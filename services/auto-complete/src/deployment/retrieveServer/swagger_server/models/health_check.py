# coding: utf-8

from __future__ import absolute_import

from datetime import date, datetime  # noqa: F401
from typing import Dict, List  # noqa: F401

from swagger_server import util
from swagger_server.models.base_model_ import Model


class HealthCheck(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self):  # noqa: E501
        """HealthCheck - a model defined in Swagger"""
        self.swagger_types = {}

        self.attribute_map = {}

    @classmethod
    def from_dict(cls, dikt) -> "HealthCheck":
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The healthCheck of this HealthCheck.  # noqa: E501
        :rtype: HealthCheck
        """
        return util.deserialize_model(dikt, cls)
