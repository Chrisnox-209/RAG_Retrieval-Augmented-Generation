from langchain_core.documents.base import Document
from langchain_text_splitters import MarkdownTextSplitter


def chunk_markdown(chunk_size: int, chunk_overlap_percent: int, data: str
                   ) -> list[tuple[int, int]]:
    if chunk_overlap_percent < 0 or chunk_overlap_percent >= 100:
        raise ValueError("The chunk_overlap_percent must be between "
                         "0 and 100.")
    if chunk_size < 1 or chunk_size > 2000:
        raise ValueError("The chunk_size must be between "
                         "1 and 2000.")

    if data == "":
        return []

    chunk_overlap = int(chunk_size * chunk_overlap_percent / 100)

    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=False,
        keep_separator=True,
    )

    documents: list[Document] = splitter.create_documents([data])

    index_list: list[tuple[int, int]] = []

    for document in documents:
        start: int = document.metadata["start_index"]
        end: int = start + len(document.page_content)

        if data[start:end] != document.page_content:
            raise ValueError("Chunk does not match the original text.")

        index_list.append((start, end))

    return index_list


def chunk_python() -> None:
    pass


def chunk_file() -> None:
    pass


def chunk_files() -> None:
    pass


if __name__ == "__main__":
    markdown_text = """
    # Guide de randonnée
    Préparer une randonnée demande une organisation rigoureuse
    pourassurer sécurité et confort

    ## Équipement essentiel
    - Veste imperméable respirante (Gore-Tex)
    - Chaussures de marche rodées
    - Gourde ou poche à eau (2L minimum)
    - Trousse de secours compacte

    ## Météo et sécurité
    La météo en montagne change très vite
    > Conseil : Consultez les prévisions locales le matin même

    1. Prévenez un proche de votre itinéraire
    2. Restez sur les sentiers balisés
    3. Sachez faire demi-tour en cas de mauvais temps

    ## Conclusion
    Une bonne préparation garantit une superbe expérience
    Restez prudent
    Pour manger des pommes
    """

    try:
        print(chunk_markdown(10, 10, "01234567890123456789"))
    except ValueError as error:
        print(f"[ERROR]: {error}")

    data = "A Bonjour \n\ntout le monde"
    position = data.rfind("\n\n", 0, 15)