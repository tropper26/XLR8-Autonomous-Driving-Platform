from PyQt5.QtWidgets import (
    QTabWidget,
    QWidget,
    QTabBar,
)

from dto.color_theme import ColorTheme


class DynamicTabWidget(QTabWidget):
    """
    A custom tab widget that supports dynamic addition and removal of tabs with a "+" tab
    that when clicked, adds a new tab using a provided factory method to generate the content.

    Attributes:
        tabs_factory (callable): A factory method provided by the parent that returns a QWidget.
                                This method is used to generate the content for new tabs.
        tabs_name (str): The name of the tabs to be displayed.
        color_theme (ColorTheme): The color theme used to style the tab widget.
    """

    def __init__(
        self,
        tabs_factory: callable,
        tabs_name: str,
        color_theme: ColorTheme,
        parent=None,
    ):
        super(DynamicTabWidget, self).__init__(parent)
        self.tabs_factory = tabs_factory
        self.tabs_name = tabs_name
        self.color_theme = color_theme
        self.addTab(QWidget(), "+")  # Adding "+" tab
        self.setTabsClosable(True)  # Enable close buttons
        self.tabCloseRequested.connect(self.close_tab)
        self.tabBarClicked.connect(self.check_tab)
        self.tabBar().setTabButton(
            self.count() - 1, QTabBar.RightSide, None
        )  # Disable close button on "+"
        self.add_new_tab()  # Add the initial tab
        self.update_tab_close_buttons()

    def add_new_tab(self):
        new_tab = self.tabs_factory()

        # Insert new tab just before the last "+" tab
        index = self.count() - 1
        self.insertTab(index, new_tab, f"{self.tabs_name} {index}")
        self.setCurrentIndex(index)
        self.update_tab_close_buttons()

    def check_tab(self, index):
        # If "+" tab is clicked, add a new tab
        if self.tabText(index) == "+":
            self.add_new_tab()

    def close_tab(self, index):
        self.removeTab(index)
        self.update_tab_close_buttons()
        if self.currentIndex() == self.count() - 1:  # If the "+" tab would be selected
            self.setCurrentIndex(self.currentIndex() - 1)  # Select the last tab instead

    def update_tab_close_buttons(self):
        # Update close button visibility based on the number of tabs
        tab_count = self.count() - 1  # Exclude the "+" tab
        if tab_count > 1:
            self.setTabsClosable(True)  # Show close buttons
        else:
            self.setTabsClosable(False)  # Hide close buttons

        self.tabBar().setTabButton(
            self.count() - 1, QTabBar.RightSide, None
        )  # Disable close button on "+"