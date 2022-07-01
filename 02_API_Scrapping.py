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

##################################################
# Functions collecting data from Yelp API
##################################################


def get_business(location, limit=50, offset=0, verbose=0):
    """
    This function connect the Yelp API and gather informations
    about businesses given the provided location parameter.

    Parameters
    ----------
    location: str
         The geographic area to be used when searching for businesses
    limit: int
        Number of business results to return
        (50 is the default AND maximum value)
    offset: int
        Offset the list of returned business results by this amount
    verbose: int
        0 -> Silent / Progress bar
        1 -> Infos about the downloaded reviews
        2 -> HTTP status

    Returns
    -------
    dictionary:
        The request result
    int:
        The updated offset
    """

    url = "https://api.yelp.com/v3/businesses/search"
    query = {
        "location": location,
        "limit": limit,
        "offset": offset,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers, params=query)
    if verbose > 2:
        print(f"HTTP status: {r.status_code} | verbose={verbose}")

    return r, offset + limit


def get_business_id(request, limit, business_ids, verbose=0):
    """
    Parse the businesses JSON collected with 'get_business()'

    Parameters
    ----------
    request: dictionnary
        A dictionary containing the businesses informations
    limit: int
        The number of business results to parse
    business_ids: list
        An external list used to collect the business ids
    verbose: int
        0 -> Silent / Progress bar
        1 -> Infos about the downloaded reviews
        2 -> HTTP status

    Returns
    -------
    It won't return anything, but the business_ids is updated
    internally, and hence externaly as this is a reference.
    """

    json = request.json()

    for i in range(limit):
        try:
            business_id = json["businesses"][i]["id"]
            business_ids.append(business_id)
            if verbose > 1:
                print(len(business_ids), business_id)

        except Exception:
            raise Exception


def get_business_reviews(id, reviews, max_reviews, collect_all=False, verbose=0):
    """
    This function connect the Yelp API and gather REVIEWS from
    the provided business ID.

    Parameters
    ----------
    id: int
        Unique Yelp ID of the business we want to get reviews from.
    reviews: list
        An external list used to collect the reviews
    max_reviews: int
        Number of business reviews to return
        (the maximum value is 200)
    collect_all: bool
        Whether or not the script should collet all ratings
    verbose: int
        0 -> Silent / Progress bar
        1 -> Infos about the downloaded reviews
        2 -> HTTP status

    Returns
    -------
    It won't return anything, but the reviews is updated
    internally, and hence externaly as this is a reference.
    """

    if collect_all:
        selected_stars = [1,2,3,4,5]
    else:
        selected_stars = [1,2]
    url = f"https://api.yelp.com/v3/businesses/{id}/reviews"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers)
    if verbose > 2:
        print(f"HTTP status: {r.status_code}")

    json = r.json()

    for review in json["reviews"]:
        text = review["text"]
        rate = review["rating"]

        if rate not in selected_stars:
            continue

        reviews.append({"text": text, "rating": rate})

        if verbose > 0:
            print(f"Num:{len(reviews)} Text:{review['text']} Rating:{review['rating']}")
        else:
            progress_monitor.items_update(
                len(reviews), max_reviews, f"Rating: {rate} Text: {text}"
            )

        if len(reviews) >= max_reviews:
            raise Exception()


def get_reviews(location, num_reviews, collect_all=False, verbose=0):
    """
    This function organize the collect of the reviews.

    Parameters
    ----------
    location: str
        The geographic area to be used when searching for businesses
    num_reviews: int
        The number of reviews to collect
    collect_all: bool
        Whether or not the script should collet all ratings
    verbose: int
        0 -> Silent / Progress bar
        1 -> Infos about the downloaded reviews
        2 -> HTTP status

    Returns
    -------
    list:
        The list of the reviews (text + rating)
    """

    business_ids = []
    business_offset = 0
    business_limit = 50
    reviews = []

    while len(reviews) <= num_reviews:
        request, business_offset = get_business(
            location,
            limit=business_limit,
            offset=business_offset,
            verbose=verbose,
        )
        try:
            get_business_id(request, business_limit, business_ids, verbose)

            for business_id in business_ids[
                business_offset - business_limit: business_offset
            ]:
                get_business_reviews(business_id, reviews, num_reviews, collect_all, verbose=verbose)

        except Exception:
            break

    if verbose == 0:
        progress_monitor.complete()

    return reviews


##################################################
# Generic functions | main + arguments
##################################################

def get_arguments():
    """
    Initialize the command line arguments and return
    the collected values if any...

    Returns
    -------
    dictionary:
        A dictionary containing the provided arguments
        along with their provided values
    """

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
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="If this argument is provided, the script will collect all ratings instead of just 1 and 2",
    )

    return parser.parse_args()


def main(**kwargs):
    """
    Verify the provided arguments, then use them to gather the reviews
    and finally save the reviews in a file.

    Parameters
    ----------
    kwargs: dictionary
        A dictionary containing the provided arguments
        along with their provided values
    """

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

    reviews = get_reviews(location, num_reviews, kwargs['all'], verbose=verbose)
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
    args = get_arguments()
    main(**vars(args))
