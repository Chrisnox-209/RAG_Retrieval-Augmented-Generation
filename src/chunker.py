
def chunk_markdown(chunk_size: int, chunk_overlap: int, data: str
                   ):
    index_list = []
    i: int = 1
    flag = False
    start_chunk: int = 0
    chunk_overlap: int = int(chunk_size * chunk_overlap / 100)

    for nb in range(0, len(data)):
        i += 1
        if chunk_size == nb:
            index_list.append((0, chunk_size))
            start_chunk: int = chunk_size - chunk_overlap
            i = 0
            flag = True
            print(f"index_list = {index_list}")
            continue

        if chunk_size == i and flag:
            end_chunk = nb - chunk_overlap
            index_list.append((start_chunk, end_chunk))
            start_chunk = nb - chunk_overlap
            i = 0

    return index_list


def chunk_python():
    pass


def chunk_file():
    pass


def chunk_files():
    pass


if __name__ == "__main__":
    markdown_text = """
    # Guide de randonnée
    Préparer une randonnée demande une organisation rigoureuse pour assurer sécurité et confort.

    ## Équipement essentiel
    - Veste imperméable respirante (Gore-Tex)
    - Chaussures de marche rodées
    - Gourde ou poche à eau (2L minimum)
    - Trousse de secours compacte

    ## Météo et sécurité
    La météo en montagne change très vite. 
    > Conseil : Consultez les prévisions locales le matin même.

    1. Prévenez un proche de votre itinéraire.
    2. Restez sur les sentiers balisés.
    3. Sachez faire demi-tour en cas de mauvais temps.

    ## Conclusion
    Une bonne préparation garantit une superbe expérience. Restez prudent !
    """

    print(chunk_markdown(100, 10, markdown_text))
