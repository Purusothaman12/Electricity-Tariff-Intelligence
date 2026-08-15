from src.models.section import Section, TOCEntry


def main():

    toc_entry = TOCEntry(
        section_id="6.1.1.1.1",
        title="RESIDENTIAL SERVICE",
        start_page=67
    )

    section = Section(
        section_id=toc_entry.section_id,
        title=toc_entry.title,
        start_page=toc_entry.start_page,
        end_page=68,
        source_file="Oncor_May_1_2023.pdf",
        category="NORMAL_SCHEDULE",
        text="Sample residential tariff section."
    )

    print("=" * 70)
    print("SECTION MODEL TEST")
    print("=" * 70)

    print("TOC Entry      :", toc_entry.full_title)
    print("TOC Start Page :", toc_entry.start_page)

    print("-" * 70)

    print("Section        :", section.full_title)
    print("Source File    :", section.source_file)
    print("Category       :", section.category)
    print("Start Page     :", section.start_page)
    print("End Page       :", section.end_page)
    print("Page Count     :", section.page_count)
    print("Contains Page 67:", section.contains_page(67))
    print("Contains Page 69:", section.contains_page(69))

    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()