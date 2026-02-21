"""
CRUCIBLE Demo Server

Serves the demo app + data files.
Usage: python serve.py [--port 8080]
"""

import argparse
import http.server
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({".js": "application/javascript", ".json": "application/json"})

    with http.server.HTTPServer(("", args.port), handler) as httpd:
        print(f"CRUCIBLE demo: http://localhost:{args.port}/demo/")
        print("Ctrl+C to stop")
        httpd.serve_forever()

if __name__ == "__main__":
    main()
