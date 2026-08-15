import pdfplumber

from src.models.document import Document, Page


class DocumentExtractor:
    """
    Extracts text and basic page metadata from a PDF document.
    """

    def extract(self, document: Document) -> Document:

        document.pages.clear()

        print(
            f"Extracting pages from: "
            f"{document.file_name}"
        )

        with pdfplumber.open(document.file_path) as pdf:

            for page_number, pdf_page in enumerate(
                pdf.pages,
                start=1
            ):

                text = pdf_page.extract_text() or ""

                page = Page(
                    page_number=page_number,
                    text=text,
                    metadata={
                        "width": float(pdf_page.width),
                        "height": float(pdf_page.height)
                    }
                )

                document.add_page(page)

        document.metadata["page_count"] = (
            document.page_count
        )

        print(
            f"Pages extracted: "
            f"{document.page_count}"
        )

        return document