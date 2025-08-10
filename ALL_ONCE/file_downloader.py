import aiohttp
import asyncio
from typing import Tuple, Optional

async def download_and_identify_file(doc_url: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Downloads content from a URL and determines its file type.

    Returns:
        A tuple of (content, file_type, error_message).
        On success, error_message is None.
        On failure, content and file_type are None.
    """
    print(f"  - 📥 Downloading document to verify identity...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(doc_url, timeout=90) as response:
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                    file_type = 'html'
                    print("  - Detected Content-Type: HTML page")
                else:
                    file_type = doc_url.split('?')[0].split('.')[-1].lower()
                    print(f"  - Detected file extension: .{file_type}")

                doc_content = await response.read()
                return doc_content, file_type, None

    except asyncio.TimeoutError:
        error_msg = "This file is too large to handle, please upload any other file"
        print(f"  - 🚨 TIMEOUT ERROR: {error_msg}")
        return None, None, error_msg
    except aiohttp.ClientError as e:
        error_msg = f"Could not download the document. Error: {e}"
        print(f"  - 🚨 DOWNLOAD ERROR: {error_msg}")
        return None, None, error_msg


# Testing

async def main(url: str):
    """Main function to run the async download."""
    
    content, file_type, error = await download_and_identify_file(url)

    if error:
        print(f"An error occurred: {error}")
    else:
        print("\n--- Download Successful ---")
        print(f"File Type: {file_type}")
        print(f"Content Size: {len(content)} bytes")
    
    return content, file_type


def run_download_and_identify_file(url: str):
    """
    This is the corrected synchronous wrapper function.
    It now correctly uses the 'url' parameter passed to it.
    """
    # The hardcoded URL has been removed from here.
    return asyncio.run(main(url))


# --- Example Usage ---
if __name__ == "__main__":
    # Now you can pass any URL to the function and it will work.
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    print("--- Testing PDF URL ---")
    run_download_and_identify_file(pdf_url)

    print("\n" + "="*40 + "\n")

    # Example with a different URL (HTML page)
    html_url = "http://example.com"
    print("--- Testing HTML URL ---")
    run_download_and_identify_file(html_url)