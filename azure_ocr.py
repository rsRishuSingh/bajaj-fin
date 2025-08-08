"""
This code sample shows Prebuilt Layout operations with the Azure AI Document Intelligence client library.
The async versions of the samples require Python 3.8 or later.

To learn more, please visit the documentation - Quickstart: Document Intelligence (formerly Form Recognizer) SDKs
https://learn.microsoft.com/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api?pivots=programming-language-python
"""

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import os
from dotenv import load_dotenv
load_dotenv()
"""
Remember to remove the key from your code when you're done, and never post it publicly. For production, use
secure methods to store and access your credentials. For more information, see 
https://docs.microsoft.com/en-us/azure/cognitive-services/cognitive-services-security?tabs=command-line%2Ccsharp#environment-variables-and-application-configuration
"""


# Hackrx Table Image
formUrl = "https://hackrx.blob.core.windows.net/assets/Test%20/image.png?sv=2023-01-03&spr=https&st=2025-08-04T19%3A21%3A45Z&se=2026-08-05T19%3A21%3A00Z&sr=b&sp=r&sig=lAn5WYGN%2BUAH7mBtlwGG4REw5EwYfsBtPrPuB0b18M4%3D"

document_intelligence_client  = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_OCR_ENDPOINT_URL"), credential=AzureKeyCredential(os.getenv("AZURE_OCR_API_KEY"))
)

poller = document_intelligence_client.begin_analyze_document(
    "prebuilt-layout", AnalyzeDocumentRequest(url_source=formUrl)
)
result = poller.result()

for idx, style in enumerate(result.styles):
    print(
        "Document contains {} content".format(
         "handwritten" if style.is_handwritten else "no handwritten"
        )
    )

print("----------------------------------------")


for page in result.pages:
    for line_idx, line in enumerate(page.lines):
        print(
         "...Line # {} has text content '{}'".format(
        line_idx,
        line.content.encode("utf-8")
        )
    )

    # for selection_mark in page.selection_marks:
    #     print(
    #      "...Selection mark is '{}' and has a confidence of {}".format(
    #      selection_mark.state,
    #      selection_mark.confidence
    #      )
    # )

print("----------------------------------------")


for table_idx, table in enumerate(result.tables):
    print(
        "Table # {} has {} rows and {} columns".format(
        table_idx, table.row_count, table.column_count
        )
    )
        
    for cell in table.cells:
        print(
            "...Cell[{}][{}] has content '{}'".format(
            cell.row_index,
            cell.column_index,
            cell.content.encode("utf-8"),
            )
        )

print("----------------------------------------")

