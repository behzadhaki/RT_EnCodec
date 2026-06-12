#!/usr/bin/env python3
"""Static dev server with caching disabled.

Plain `python -m http.server` sends no Cache-Control header, so browsers
heuristically cache ES modules (web/lib/*.js) and serve stale code after
edits — only hard reloads pick up changes. This variant sends
`Cache-Control: no-store` on every response so a normal reload always
fetches fresh files.

Usage: python3 devserver.py [port]   (default 8765, serves CWD)
"""
import http.server
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.test(HandlerClass=NoCacheHandler, port=port)
