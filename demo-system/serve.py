#!/usr/bin/env python3
"""Serve the Crucible pipeline demo. Open http://localhost:8081/demo-system/"""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8081

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
handler = http.server.SimpleHTTPRequestHandler
with http.server.HTTPServer(("", PORT), handler) as httpd:
    print(f"Serving at http://localhost:{PORT}/demo-system/")
    httpd.serve_forever()
