class WorldManager:
    def __init__(self):
        self.__objects = []

    @property
    def objects(self):
        return self.__objects

    def clear(self):
        self.__objects = []

    def append(self, *objects):
        for object in objects:
            self.__objects.append(object)

    def query_objects(self, name, tags = None):
        return list(
            filter(
                lambda object: object.name == name and (tags is None or object.tags == tags),
                self.objects,
            )
        )

    def format_prompt_specification(self):
        prompt = "Available objects (name [tags] (dimension; center_position)):\n"

        for object in self.__objects:
            prompt += "{name} [{tags}] ({dimension}; {center_position})\n".format(
                name=object.name,
                tags=", ".join(object.tags),
                dimension=object.dimension,
                center_position=object.position,
            )

        return prompt
