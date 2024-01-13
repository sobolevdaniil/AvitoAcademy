import io

from what_is_year_now import what_is_year_now
from unittest.mock import patch
import pytest


def test_point_time():
    with patch("what_is_year_now.urllib.request.urlopen") as mocked_get_cases:
        mocked_get_cases.return_value = io.StringIO('{"currentDateTime": "19.11.2022"}')
        assert what_is_year_now() == 2022
        mocked_get_cases.assert_called_once()


def test_blank_time():
    with patch("what_is_year_now.urllib.request.urlopen") as mocked_get_cases:
        mocked_get_cases.return_value = io.StringIO('{"currentDateTime": "2022-11-19"}')
        assert what_is_year_now() == 2022
        mocked_get_cases.assert_called_once()


def test_value_error():
    with patch("what_is_year_now.urllib.request.urlopen") as mocked_get_cases:
        mocked_get_cases.return_value = io.StringIO('{"currentDateTime": "2022.11.19"}')
        with pytest.raises(ValueError):
            what_is_year_now()


def test_empty_date():
    with patch("what_is_year_now.urllib.request.urlopen") as mocked_get_cases:
        mocked_get_cases.return_value = io.StringIO('{"currentDateTime": ""}')
        with pytest.raises(IndexError):
            what_is_year_now()
