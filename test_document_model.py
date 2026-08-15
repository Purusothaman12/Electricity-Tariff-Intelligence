from src.models.document import Document, Page


def main():

    document = Document(
        file_name="sample.pdf",
        file_path="data/sample.pdf"
    )

    document.add_page(
        Page(
            page_number=1,
            text="First page"
        )
    )

    document.add_page(
        Page(
            page_number=2,
            text="Second page"
        )
    )

    print("=" * 60)
    print("DOCUMENT MODEL TEST")
    print("=" * 60)

    print("File Name :", document.file_name)
    print("File Path :", document.file_path)
    print("Page Count:", document.page_count)

    page_two = document.get_page(2)

    if page_two is not None:
        print("Page 2 Text:", page_two.text)
    else:
        print("Page 2 was not found.")

    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()