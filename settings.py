"""Local configuration. Secrets never enter session metadata or logs."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlparse
import re
import uuid
from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / '.env'


def notion_page_id(value):
    value = value.strip()
    if '://' in value:
        parsed = urlparse(value)
        host = (parsed.hostname or '').lower()
        if parsed.scheme != 'https' or not any(host == d or host.endswith('.' + d)
                for d in ('notion.so', 'notion.site', 'notion.com')):
            raise ValueError('请填写 Notion 页面链接或页面 ID。')
        value = parsed.path.rstrip('/').split('/')[-1]
    match = re.search(r'([0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})$', value)
    if not match:
        raise ValueError('无法识别 Notion 页面 ID，请粘贴目标页面的完整链接。')
    return str(uuid.UUID(match.group(1)))


@dataclass
class NotionSettings:
    api_key: str = field(default='', repr=False)
    parent_page: str = ''
    auto_sync: bool = False

    def validate(self):
        if not self.api_key.strip():
            raise ValueError('请在 Notion 设置中填写 Internal connection Token。')
        return notion_page_id(self.parent_page)

    @classmethod
    def load(cls, path=ENV_PATH):
        values = dotenv_values(path) if Path(path).exists() else {}
        def value(name, default=''):
            return os.environ.get(name, values.get(name) or default)
        return cls(value('NOTION_API_KEY'), value('NOTION_PARENT_PAGE_ID'),
                   value('NOTION_AUTO_SYNC', 'false').lower() == 'true')

    def save(self, path=ENV_PATH):
        if self.parent_page.strip():
            notion_page_id(self.parent_page)
        path = Path(path)
        path.touch(exist_ok=True)
        for name, value in [('NOTION_API_KEY', self.api_key.strip()),
                            ('NOTION_PARENT_PAGE_ID', self.parent_page.strip()),
                            ('NOTION_AUTO_SYNC', str(self.auto_sync).lower())]:
            set_key(str(path), name, value)


@dataclass
class DeepSeekSettings:
    api_key: str = field(default='', repr=False)
    base_url: str = 'https://api.deepseek.com'
    model: str = 'deepseek-flash'
    timeout: float = 120

    def validate(self):
        if not self.api_key.strip():
            raise ValueError('请在 DeepSeek 设置中填写 API Key。')
        parsed = urlparse(self.base_url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or
                parsed.password or parsed.query or parsed.fragment):
            raise ValueError('DeepSeek API 地址必须是有效的 HTTPS 地址。')
        if not self.model.strip() or not 5 <= self.timeout <= 600:
            raise ValueError('请填写模型名；请求超时须为 5～600 秒。')

    @classmethod
    def load(cls, path=ENV_PATH):
        values = dotenv_values(path) if Path(path).exists() else {}
        def value(name, default):
            return os.environ.get(name, values.get(name) or default)
        try:
            timeout = float(value('DEEPSEEK_TIMEOUT', '120'))
        except ValueError:
            timeout = 120
        return cls(value('DEEPSEEK_API_KEY', ''), value('DEEPSEEK_BASE_URL', cls.base_url),
                   value('DEEPSEEK_MODEL', cls.model), timeout)

    def save(self, path=ENV_PATH):
        path = Path(path)
        path.touch(exist_ok=True)
        for name, value in [('DEEPSEEK_API_KEY', self.api_key.strip()),
                            ('DEEPSEEK_BASE_URL', self.base_url.strip().rstrip('/')),
                            ('DEEPSEEK_MODEL', self.model.strip()),
                            ('DEEPSEEK_TIMEOUT', str(self.timeout))]:
            set_key(str(path), name, value)
