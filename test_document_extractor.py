from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.document_extractor import DocumentExtractor


def main():

    loader = PDFLoader()
    extractor = DocumentExtractor()

    documents = loader.load("data")

    print("=" * 80)
    print("DOCUMENT EXTRACTOR TEST")
    print("=" * 80)

    for document in documents:

        document = extractor.extract(
            document
        )

        print()
        print("-" * 80)
        print("File Name :", document.file_name)
        print("Page Count:", document.page_count)

        first_page = document.get_page(1)

        if first_page is None:

            print("First page was not found.")

        else:

            preview = (
                first_page.text
                .replace("\n", " ")
                .strip()
            )

            print(
                "Page 1 Width :",
                first_page.metadata.get("width")
            )

            print(
                "Page 1 Height:",
                first_page.metadata.get("height")
            )

            print(
                "Page 1 Preview:"
            )

            print(
                preview[:500]
                if preview
                else "[No text extracted]"
            )

        print("-" * 80)

    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()