#! /usr/bin/env python3
# coding: utf-8

""" The purpose of this module is to gather
    the generic functions
"""

from os import get_terminal_size

##################################################
# Progress
##################################################


class Progress:
    """The purpose of this class is to display the current
        status of the website / API scraping

    Attributes
    ----------
    _items : dict
        Overall items progress informations

    Methods
    -------
    items_update(current, total, label)
        update the current scraping progress status
    items_init(total, label)
        initilize the overall scraping informations
    """

    def __init__(self):
        self._items = {"current": 0, "total": 0, "label": ""}
        self.error_count = 0

    def items_update(self, current, total, label):
        self._items = {
            "current": int(current),
            "total": int(total),
            "label": label,
        }

        self.__update_display()

    def items_init(self, total, label):
        self._items = {
            "current": 0,
            "total": int(total),
            "label": label,
        }

    def complete(self):

        try:
            terminal_size = get_terminal_size()
            size = terminal_size.columns - 1

            print("\033[B" * 1)

            print("\n" + " Scraping process complete ".center(size, "*"[:size]))

        except OSError:
            pass

    # --- PRIVATE METHODS ---

    def __update_display(self):

        try:
            terminal_size = get_terminal_size()
            bar_size = terminal_size.columns - 20
            num_lines = 3

            # Clean terminal
            print(" " * terminal_size.columns * num_lines)
            print("\033[A" * (num_lines + 1))

            # Display
            allitems = self._items
            all_bar = self.__get_progressbar(allitems, bar_size)
            if self.error_count == 0:
                title = f"{allitems['label']}"
            else:
                title = f"{allitems['label']} [There are {self.error_count} error(s) : check errors.log]"

            clean_title = title.replace("\n", " ")
            print(f"{clean_title[:bar_size].ljust(bar_size)}")
            print(f"{all_bar} {allitems['current']}/{allitems['total']} items")

            # Reset cursor position in terminal
            print("\033[A" * num_lines)

        except OSError:
            pass

    def __get_progressbar(self, source, bar_size):

        todo_char = "◻"
        done_char = "◼"

        try:
            size = round(bar_size / source["total"] * source["current"])
        except ZeroDivisionError:
            size = 0

        fillchars = done_char * size
        return f"{fillchars.ljust(bar_size,todo_char)}"


progress_monitor = Progress()
