from pathlib import Path

from src.models.document import Document


class PDFLoader:
    """
    Finds PDF files inside a directory and creates Document objects.
    """

    def load(self, directory_path: str) -> list[Document]:

        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {directory}"
            )

        pdf_files = sorted(
            directory.glob("*.pdf"),
            key=lambda path: path.name.lower()
        )

        documents = []

        for pdf_path in pdf_files:

            document = Document(
                file_name=pdf_path.name,
                file_path=str(pdf_path.resolve())
            )

            documents.append(document)

        return documents