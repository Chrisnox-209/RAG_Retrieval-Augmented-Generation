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

    if len(data) <= chunk_size:
        return [(0, len(data))]

    index_list: list[tuple[int, int]] = []
    start_chunk: int = 0
    chunk_overlap = int(chunk_size * chunk_overlap_percent / 100)

    while start_chunk < len(data):

        end_chunk: int = min(start_chunk + chunk_size, len(data))

        index_list.append((start_chunk, end_chunk))

        if end_chunk == len(data):
            break

        start_chunk = start_chunk + chunk_size - chunk_overlap

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

    print(position)  # 1