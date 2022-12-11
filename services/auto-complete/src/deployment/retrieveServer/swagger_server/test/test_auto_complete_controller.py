# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.health_check import HealthCheck  # noqa: E501
from swagger_server.models.prefix_retrieve import PrefixRetrieve  # noqa: E501
from swagger_server.test import BaseTestCase


class TestAutoCompleteController(BaseTestCase):
    """AutoCompleteController integration test stubs"""

    def test_find_completions_by_charater(self):
        """Test case for find_completions_by_charater

        Finds completion candidates
        """
        body = 'body_example'
        response = self.client.open(
            '/v1/prefixRetrieve',
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_health_check(self):
        """Test case for health_check

        Health check
        """
        response = self.client.open(
            '/v1/healthCheck',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
