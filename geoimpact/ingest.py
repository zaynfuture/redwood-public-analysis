"""Bounded public-news ingestion. Run locally, behind restricted egress if deployed."""
import hashlib
import ipaddress
import socket
from io import BytesIO
from urllib.parse import urlsplit, urljoin
import requests
from bs4 import BeautifulSoup

MAX_BYTES = 8 * 1024 * 1024
MAX_CHARS = 24000


def validate_url(url: str):
    parts = urlsplit(url)
    if parts.scheme not in ('https', 'http') or not parts.hostname or parts.username or parts.password:
        raise ValueError('Use a public HTTP(S) news URL without credentials.')
    if parts.port not in (None, 80, 443):
        raise ValueError('Only standard web ports are supported.')
    addresses = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == 'https' else 80))
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError('Private, local, and reserved network destinations are blocked.')


def pdf_text(content: bytes) -> str:
    from pypdf import PdfReader
    if len(content) > MAX_BYTES:
        raise ValueError('PDF exceeds the 8 MB limit.')
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted or len(reader.pages) > 40:
        raise ValueError('Use an unencrypted PDF of at most 40 pages.')
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    if len(text.strip()) < 50:
        raise ValueError('No usable PDF text. Scanned PDFs require OCR before upload.')
    return text


def fetch(url: str) -> tuple[str, str]:
    session = requests.Session()
    session.trust_env = False
    for _ in range(4):
        validate_url(url)
        with session.get(url, timeout=(8, 20), stream=True, allow_redirects=False,
                         headers={'User-Agent': 'RedwoodPublicAnalysis/0.1'}) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers['Location'])
                continue
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    raise ValueError('Source exceeds the 8 MB download limit.')
            content = bytes(body)
            kind = response.headers.get('Content-Type', '').lower()
            if content.startswith(b'%PDF'):
                return pdf_text(content), url
            if 'html' not in kind and 'text/plain' not in kind:
                raise ValueError('This URL did not return a supported HTML, text, or PDF document.')
            soup = BeautifulSoup(content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            main = soup.find('article') or soup.find('main') or soup
            blocks = main.find_all(['h1', 'p'])
            # Preserve article paragraphs while excluding share buttons and image credits.
            text = '\n'.join(block.get_text(' ', strip=True) for block in blocks) if main.find('p') else main.get_text(' ', strip=True)
            return text, url
    raise ValueError('Too many redirects.')


def prepare(text: str, origin: str) -> dict:
    text = ' '.join(text.split())
    if len(text) < 50:
        raise ValueError('Please provide at least 50 characters of readable news text.')
    return {'id': 'source-' + hashlib.sha256(text.encode()).hexdigest()[:16],
            'origin': origin, 'text': text[:MAX_CHARS], 'truncated': len(text)>MAX_CHARS,
            'original_characters': len(text)}
