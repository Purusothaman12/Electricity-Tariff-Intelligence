from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.document_extractor import DocumentExtractor
from src.parsing.toc_parser import TOCParser
from src.parsing.schedule_classifier import ScheduleClassifier


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    classifier = ScheduleClassifier()

    documents = loader.load("data")

    print("=" * 100)
    print("SCHEDULE CLASSIFIER TEST")
    print("=" * 100)

    for document in documents:

        document = document_extractor.extract(
            document
        )

        entries = toc_parser.parse(
            document
        )

        classified = classifier.classify_all(
            entries
        )

        print()
        print("=" * 100)
        print(document.file_name)
        print("=" * 100)

        for category, category_entries in classified.items():

            print()
            print(
                f"{category}: "
                f"{len(category_entries)}"
            )

            print("-" * 100)

            for entry in category_entries:

                normalized_name = (
                    classifier.normalize_schedule_name(
                        entry.title
                    )
                )

                print(
                    f"{entry.section_id:<15} "
                    f"{entry.title:<65} "
                    f"Normalized: {normalized_name}"
                )

    print()
    print("=" * 100)
    print("TEST COMPLETED")
    print("=" * 100)


if __name__ == "__main__":
    main()