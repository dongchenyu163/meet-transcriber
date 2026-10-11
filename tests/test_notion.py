import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import httpx
from notion_sync import NotionSync, NotionError, sync_lock
from settings import NotionSettings, notion_page_id

PARENT = 'a1234567-1234-1234-1234-123456789abc'
CHILD = 'b1234567-1234-1234-1234-123456789abc'


class Server:
    def __init__(self):
        self.title = None
        self.markdown = ''
        self.calls = []
        self.lose_create = False
        self.lose_patch = False

    def __call__(self, request):
        self.calls.append((request.method, request.url.path))
        assert request.headers['notion-version'] == '2026-03-11'
        assert str(request.url).startswith('https://api.notion.com/v1/')
        if request.method == 'GET' and request.url.path == '/v1/pages/' + PARENT:
            return httpx.Response(200, json={'object':'page', 'id':PARENT})
        if request.url.path.endswith('/children'):
            return httpx.Response(200, json={'results': ([{'type':'child_page','id':CHILD,
                'child_page':{'title':self.title}}] if self.title else []), 'has_more':False})
        if request.method == 'POST':
            body = json.loads(request.content)
            assert body['parent']['page_id'] == PARENT
            self.title = body['properties']['title']['title'][0]['text']['content']
            if self.lose_create:
                self.lose_create = False
                raise httpx.ReadTimeout('echo secret', request=request)
            return httpx.Response(200, json={'id':CHILD})
        if request.url.path == '/v1/pages/' + CHILD:
            return httpx.Response(200, json={'id':CHILD,'parent':{'page_id':PARENT}})
        if request.method == 'PATCH':
            self.markdown = json.loads(request.content)['replace_content']['new_str']
            if self.lose_patch:
                self.lose_patch = False
                raise httpx.ReadTimeout('lost response', request=request)
        return httpx.Response(200, json={'markdown':self.markdown,'truncated':False})


class NotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='meet-notion-')
        self.root = Path(self.temp.name)
        (self.root/'minutes.md').write_text('# 纪要\n\n- 行动项\n',encoding='utf-8')
        self.server = Server()
        self.settings = NotionSettings('test-secret', PARENT)
        self.client = NotionSync(self.settings, transport=httpx.MockTransport(self.server))

    def tearDown(self): self.temp.cleanup()

    def test_create_then_repeat_then_update_same_child(self):
        url = self.client.sync(self.root)
        self.assertIn(CHILD.replace('-',''),url)
        self.assertEqual(self.server.markdown, '# 纪要\n\n- 行动项\n')
        self.client.sync(self.root)
        (self.root/'minutes.md').write_text('# 新纪要',encoding='utf-8')
        self.client.sync(self.root)
        self.assertEqual(self.server.markdown,'# 新纪要')
        self.assertEqual(sum(m=='POST' for m,_ in self.server.calls),1)
        self.assertEqual(sum(m=='PATCH' for m,_ in self.server.calls),2)
        self.assertNotIn('test-secret',(self.root/'notion-sync.json').read_text())

    def test_lost_create_response_reconciles_without_duplicate(self):
        self.server.lose_create = True
        with self.assertRaises(NotionError): self.client.sync(self.root)
        self.client.sync(self.root)
        self.assertEqual(sum(m=='POST' for m,_ in self.server.calls),1)
        self.assertTrue(self.server.markdown)

    def test_lost_patch_response_recognizes_already_applied_content(self):
        self.server.lose_patch = True
        with self.assertRaises(NotionError): self.client.sync(self.root)
        self.client.sync(self.root)
        self.assertEqual(sum(m=='PATCH' for m,_ in self.server.calls),1)

    def test_remote_manual_edit_is_preserved_even_without_local_change(self):
        self.client.sync(self.root)
        self.server.markdown += '\n手动补充'
        with self.assertRaisesRegex(NotionError, '保留远端'): self.client.sync(self.root)
        (self.root/'minutes.md').write_text('new',encoding='utf-8')
        with self.assertRaises(NotionError): self.client.sync(self.root)
        self.assertIn('手动补充',self.server.markdown)

    def test_auth_error_redacts_body_and_keeps_local_minutes(self):
        client = NotionSync(self.settings, transport=httpx.MockTransport(lambda r: httpx.Response(401,text='test-secret')))
        with self.assertRaises(NotionError) as error: client.sync(self.root)
        self.assertNotIn('test-secret',str(error.exception))
        self.assertTrue((self.root/'minutes.md').exists())

    def test_network_cancellation_is_prompt(self):
        import time
        cancel = threading.Event()
        async def handle(request):
            cancel.set()
            await asyncio.sleep(10)
        client = NotionSync(self.settings, cancel=cancel, transport=httpx.MockTransport(handle))
        started = time.monotonic()
        with self.assertRaisesRegex(NotionError,'取消'): client.check_connection()
        self.assertLess(time.monotonic()-started,2)

    def test_parent_read_requires_no_write(self):
        self.client.check_connection()
        self.assertEqual(self.server.calls,[('GET','/v1/pages/'+PARENT)])

    def test_changed_speaker_names_require_regenerating_minutes(self):
        (self.root/'.minutes-stale').write_text('changed')
        with self.assertRaisesRegex(NotionError,'重新生成'): self.client.sync(self.root)
        self.assertFalse(self.server.calls)

    def test_lock_prevents_concurrent_duplicate_creation(self):
        with sync_lock(self.root):
            with self.assertRaisesRegex(NotionError,'另一个进程'): self.client.sync(self.root)
        self.assertFalse(self.server.calls)

    def test_parse_link_ignores_view_and_block_ids(self):
        self.assertEqual(notion_page_id('https://www.notion.so/workspace/Notes-'+PARENT.replace('-','')+'?v='+CHILD+'#'+CHILD),PARENT)
        for invalid in ('https://evil.test/'+PARENT, 'https://notion.so.evil.test/'+PARENT, 'abc'):
            with self.assertRaises(ValueError): notion_page_id(invalid)

    def test_config_does_not_overwrite_deepseek_or_leak_secret_repr(self):
        path = self.root/'.env'
        path.write_text('DEEPSEEK_API_KEY=keep-this\n',encoding='utf-8')
        self.settings.save(path)
        with patch.dict('os.environ',{},clear=True): restored = NotionSettings.load(path)
        self.assertEqual(restored.api_key,'test-secret')
        self.assertNotIn('test-secret',repr(restored))
        self.assertIn('DEEPSEEK_API_KEY=keep-this',path.read_text())


if __name__ == '__main__': unittest.main()
