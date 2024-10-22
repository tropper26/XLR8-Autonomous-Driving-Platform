class ColorTheme:
    def __init__(
        self,
        primary_color: str,
        secondary_color: str,
        selected_color: str,
        background_color: str,
    ):
        self.primary_color = primary_color
        self.background_color = background_color
        self.secondary_color = secondary_color
        self.selected_color = selected_color
        self.hover_color = "#e02f35"
        self.text_color = "white"
        self.button_text_color = self.text_color