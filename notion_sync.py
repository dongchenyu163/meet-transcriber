"""Notion internal connection: one managed child page per local meeting."""
import asyncio
from contextlib import contextmanager, suppress
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import threading

import httpx
from settings import notion_page_id
from storage import atomic_write

API = 'https://api.notion.com/v1'
VERSION = '2026-03-11'


class NotionError(RuntimeError):
    def __init__(self, message, uncertain=False):
        super().__init__(message)
        self.uncertain = uncertain


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@contextmanager
def sync_lock(root):
    with (root / '.notion-sync.lock').open('a+b') as file:
        if file.tell() == 0:
            file.write(b'0')
            file.flush()
        file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise NotionError('这场会议正在另一个进程中同步，请稍后重试。') from None
        try:
            yield
        finally:
            file.seek(0)
            if os.name == 'nt':
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file, fcntl.LOCK_UN)


class NotionSync:
    def __init__(self, settings, cancel=None, progress=None, transport=None):
        self.settings = settings
        self.cancel = cancel or threading.Event()
        self.progress = progress or (lambda _: None)
        self.transport = transport

    async def _send(self, method, path, **kwargs):
        async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=15),
                transport=self.transport, follow_redirects=False) as client:
            task = asyncio.create_task(client.request(method, API + path,
                headers={'Authorization': 'Bearer ' + self.settings.api_key,
                         'Notion-Version': VERSION}, **kwargs))
            try:
                async with asyncio.timeout(60):
                    while not task.done():
                        if self.cancel.is_set():
                            raise NotionError('已取消 Notion 同步；本地纪要已保留。', uncertain=True)
                        await asyncio.wait({task}, timeout=.1)
                    return await task
            finally:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

    def request(self, method, path, **kwargs):
        for attempt in range(3):
            if self.cancel.is_set():
                raise NotionError('已取消 Notion 同步。')
            try:
                response = asyncio.run(self._send(method, path, **kwargs))
            except (httpx.HTTPError, TimeoutError):
                raise NotionError('Notion 连接失败或超时，未自动重复写入；请重试以核对结果。',
                                  uncertain=method != 'GET') from None
            if (response.status_code == 429 or (method == 'GET' and response.status_code >= 500)) and attempt < 2:
                try:
                    delay = min(60, max(1, float(response.headers.get('Retry-After', 2 ** attempt))))
                except ValueError:
                    delay = 2 ** attempt
                if self.cancel.wait(delay):
                    raise NotionError('已取消 Notion 同步。')
                continue
            if not 200 <= response.status_code < 300:
                hints = {400: '内容或页面格式不支持，请检查页面和纪要大小',
                         401: 'Token 无效', 403: '请开启读取、插入和更新内容权限',
                         404: '页面不存在，或尚未共享给这个内部连接',
                         429: '请求过于频繁，请稍后重试'}
                raise NotionError(f'Notion 请求失败（HTTP {response.status_code}）：' +
                    hints.get(response.status_code, '服务暂时不可用，请稍后核对结果'),
                    uncertain=response.status_code >= 500 or response.status_code in (408, 409))
            try:
                data = response.json()
                if not isinstance(data, dict) or response.status_code == 202:
                    raise ValueError()
                return data
            except ValueError:
                raise NotionError('Notion 响应格式无效，请稍后核对结果。', uncertain=True) from None

    def check_connection(self):
        parent = self.settings.validate()
        page = self.request('GET', '/pages/' + parent)
        if page.get('archived') or page.get('in_trash'):
            raise NotionError('目标页面已归档或在回收站中。')
        return page

    def _find_child(self, parent, title):
        params = {'page_size': 100}
        found = []
        while True:
            data = self.request('GET', f'/blocks/{parent}/children', params=params)
            found.extend(b['id'] for b in data.get('results', [])
                         if b.get('type') == 'child_page' and b['child_page'].get('title') == title)
            if not data.get('has_more'):
                break
            cursor = data.get('next_cursor')
            if not cursor or cursor == params.get('start_cursor'):
                raise NotionError('Notion 分页响应无效。')
            params['start_cursor'] = cursor
        if len(found) > 1:
            raise NotionError('发现多个同名会议页面，请在 Notion 中核对重复页面。')
        return notion_page_id(found[0]) if found else None

    def sync(self, session_dir):
        parent = self.settings.validate()
        root = Path(session_dir)
        if (root / '.minutes-stale').exists():
            raise NotionError('说话人姓名已修改，请先重新生成会议纪要再同步。')
        markdown = (root / 'minutes.md').read_text(encoding='utf-8')
        if not markdown.strip():
            raise NotionError('本地纪要为空，请先生成纪要。')
        if len(markdown.encode('utf-8')) > 350_000:
            raise NotionError('纪要超过单次上传大小，请先缩短到 350KB 以内。')
        with sync_lock(root):
            state_path = root / 'notion-sync.json'
            state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
            entry = state.setdefault(parent, {})
            def save():
                atomic_write(state_path, json.dumps(state, ensure_ascii=False, indent=2))
            self.check_connection()
            page_id = entry.get('page_id')
            if not page_id:
                # Unique meeting suffix lets us reconcile a lost creation response.
                entry.setdefault('title', '会议纪要 · ' + root.name[:150])
                page_id = self._find_child(parent, entry['title'])
                if not page_id:
                    if entry.get('pending_create'):
                        raise NotionError('上次创建结果尚未确认，暂未找到页面。请稍后重试，避免重复上传。')
                    entry['pending_create'] = True
                    save()
                    self.progress('正在 Notion 目标页面下创建会议纪要…')
                    try:
                        page = self.request('POST', '/pages', json={
                            'parent': {'page_id': parent},
                            'properties': {'title': {'type': 'title', 'title': [
                                {'type': 'text', 'text': {'content': entry['title']}}]}}})
                        page_id = notion_page_id(page.get('id', ''))
                    except NotionError as exc:
                        if not exc.uncertain:
                            entry.pop('pending_create', None)
                            save()
                        raise
                entry['page_id'] = page_id
                entry.pop('pending_create', None)
                save()
            page_id = notion_page_id(page_id)
            # Check parent to avoid following a moved page into a different project.
            page = self.request('GET', '/pages/' + page_id)
            if page.get('archived') or page.get('in_trash') or page.get('parent', {}).get('page_id') != parent:
                raise NotionError('已同步的会议页面被移动或归档，请先在 Notion 中核对。')
            remote = self.request('GET', f'/pages/{page_id}/markdown')
            if remote.get('truncated') or not isinstance(remote.get('markdown'), str):
                raise NotionError('无法完整读取 Notion 页面，已停止更新以保留现有内容。')
            remote_hash = digest(remote['markdown'])
            if not entry.get('remote_hash'):
                if remote['markdown'].strip():
                    raise NotionError('找到了已有会议页面，但缺少同步基准，已停止自动覆盖。请核对 Notion 页面。')
                entry['remote_hash'] = remote_hash
                save()
            if entry['remote_hash'] != remote_hash:
                if entry.get('pending_hash') == digest(markdown) and remote['markdown'].strip() == markdown.strip():
                    entry['uploaded_hash'] = digest(markdown)
                else:
                    raise NotionError('Notion 页面已被修改或上次更新结果不确定，已保留远端内容。请核对后再同步。')
            # A retry after a lost PATCH response can recognize the resulting content.
            if entry.get('uploaded_hash') != digest(markdown):
                if remote['markdown'].strip() != markdown.strip():
                    self.progress('正在更新这场会议对应的 Notion 子页面…')
                    entry['pending_hash'] = digest(markdown)
                    save()
                    result = self.request('PATCH', f'/pages/{page_id}/markdown', json={
                        'type': 'replace_content', 'replace_content': {'new_str': markdown}})
                    if not isinstance(result.get('markdown'), str) or result.get('truncated'):
                        raise NotionError('更新响应不完整，请稍后核对结果。', uncertain=True)
                    remote_hash = digest(result['markdown'])
                entry['uploaded_hash'] = digest(markdown)
            entry['remote_hash'] = remote_hash
            entry.pop('pending_hash', None)
            entry['url'] = 'https://www.notion.so/' + page_id.replace('-', '')
            entry['synced_at'] = datetime.now().isoformat(timespec='seconds')
            save()
            return entry['url']
