#!/usr/bin/env python
# coding: utf-8


import os
import requests
import argparse

import pandas as pd

from utils import progress_monitor

# IMPORT Yelp API Key
with open("secrets.txt") as f:
    API_KEY = f.readline().strip()


def get_business(search, limit=50, offset=0, verbose=0):
    """TODO"""

    url = "https://api.yelp.com/v3/businesses/search"
    query = {
        "location": search,
        "limit": limit,
        "offset": offset,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers, params=query)
    if verbose > 2:
        print(f"HTTP status: {r.status_code} | verbose={verbose}")

    return r, offset + limit


def get_business_id(request, limit, business_ids, verbose=0):
    """TODO"""

    json = request.json()

    for i in range(limit):
        try:
            business_id = json["businesses"][i]["id"]
            business_ids.append(business_id)
            if verbose > 1:
                print(len(business_ids), business_id)

        except Exception:
            raise Exception


def get_business_reviews(id, reviews, max_reviews, verbose=0):
    """TODO"""

    url = f"https://api.yelp.com/v3/businesses/{id}/reviews"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers)
    if verbose > 2:
        print(f"HTTP status: {r.status_code}")

    json = r.json()

    for review in json["reviews"]:
        text = review["text"]
        rate = review["rating"]
        reviews.append({"text": text, "rating": rate})

        if verbose > 0:
            print(f"Num:{len(reviews)} Text:{review['text']} Rating:{review['rating']}")
        else:
            progress_monitor.items_update(
                len(reviews), max_reviews, f"Rating: {rate} Text: {text}"
            )

        if len(reviews) >= max_reviews:
            raise Exception()


def get_reviews(business_location, num_reviews, verbose=0):
    """TODO"""

    business_ids = []
    business_offset = 0
    business_limit = 50
    reviews = []

    while len(reviews) <= num_reviews:
        request, business_offset = get_business(
            business_location,
            limit=business_limit,
            offset=business_offset,
            verbose=verbose,
        )
        try:
            get_business_id(request, business_limit, business_ids, verbose)

            for business_id in business_ids[
                business_offset - business_limit : business_offset
            ]:
                get_business_reviews(business_id, reviews, num_reviews, verbose=verbose)

        except Exception:
            break

    if verbose == 0:
        progress_monitor.complete()

    return reviews


# def main(location=None, num_reviews=None, save_path=None, verbose=None):
def main(**kwargs):

    if kwargs["location"] is None:
        location = "France"
    else:
        location = kwargs["location"]

    if kwargs["num_reviews"] is None:
        num_reviews = 200
    else:
        num_reviews = int(kwargs["num_reviews"])

    if kwargs["save_path"] is None:
        save_path = os.path.join("data", "api_export.csv")
    else:
        save_path = kwargs["save_path"]

    if kwargs["verbose"] is None:
        verbose = 0
    else:
        verbose = int(kwargs["verbose"])

    print(
        f"location:{location}, num_reviews:{num_reviews}, save_path:{save_path}, verbose:{verbose}\n"
    )

    if verbose == 0:
        progress_monitor.items_init(num_reviews, "Connecting to API...")

    reviews = get_reviews(location, num_reviews, verbose=verbose)
    reviews_df = pd.DataFrame(reviews)
    reviews_df.to_csv(save_path, index=False)

    terminal_size = os.get_terminal_size()
    size = terminal_size.columns - 1
    print(
        "\n"
        + f" Exporting dataset of shape {reviews_df.shape} to {save_path} ".center(
            size, "*"[:size]
        ),
        "\n",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_reviews",
        # action="store_true",
        help="The number of reviews to collect",
    )
    parser.add_argument(
        "-p",
        "--save_path",
        # action="store_true",
        help="The path to the CSV file (including the extension)",
    )
    parser.add_argument(
        "-l",
        "--location",
        # action="store_true",
        help="The location name where the reviews will be collected",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        # action="store_true",
        help="The verbosity level (0:Nothing | 1:Reviews | 2:+Business_ids | 3:+HTTP status",
    )

    args = parser.parse_args()

    main(**vars(args))
