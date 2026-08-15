from src.ingestion.pdf_loader import PDFLoader


def main():

    loader = PDFLoader()

    documents = loader.load("data")

    print("=" * 70)
    print("PDF LOADER TEST")
    print("=" * 70)

    print("PDF files found:", len(documents))
    print()

    if not documents:
        print("No PDF files were found inside the data folder.")

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(f"Document {index}")
        print("File Name :", document.file_name)
        print("File Path :", document.file_path)
        print("Page Count:", document.page_count)
        print("-" * 70)

    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()