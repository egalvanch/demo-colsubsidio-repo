import os
import aiohttp
import ssl
from azure.core.pipeline.transport import AioHttpTransport

class CustomAzureTransport(AioHttpTransport):
    def __init__(self, *args, **kwargs):
        print("[INFO] CustomAzureTransport initialized")
        super().__init__(*args, **kwargs)

    async def __aenter__(self):
        if not hasattr(self, 'session'):
            dev_mode = os.getenv("ENVIRONMENT", "development") != "production"
            if dev_mode:
                ssl_context = ssl._create_unverified_context()
                print("[DEBUG] SSL verification disabled (dev mode)")
            else:
                ssl_context = ssl.create_default_context(cafile=os.getenv("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt"))
                print("[DEBUG] SSL verification enabled with cert:", ssl_context)

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
        return self
