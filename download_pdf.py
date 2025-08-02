import requests


def download_pdf(url: str, save_path: str = "temp.pdf") -> bool:
    """Downloads a PDF from a URL and saves it locally."""
    try:
        print(f"⬇️  Downloading PDF from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        print(f"✅ PDF saved successfully to '{save_path}'")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download PDF: {e}")
        return False