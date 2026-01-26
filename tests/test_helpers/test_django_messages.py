from unittest.mock import patch

from django.contrib.auth.models import User
from django.db import connection, connections
from django.db.backends.sqlite3.features import DatabaseFeatures

import pytest
from langchain_core.messages import HumanMessage
from model_bakery import baker

from django_ai_assistant.helpers.django_messages import save_django_messages
from django_ai_assistant.models import Message, Thread


@pytest.mark.django_db()
def test_django_messages_with_can_return_rows_from_bulk_insert_true():
    class MockFeatures(DatabaseFeatures):
        can_return_rows_from_bulk_insert = True

    mock_features = MockFeatures(connections[Message.objects.db])

    thread = baker.make(Thread, created_by=baker.make(User))
    with patch.object(
        Message.objects,
        "bulk_create",
        wraps=Message.objects.bulk_create,
    ) as mock_bulk_create:
        with patch.object(connection, "features", mock_features):
            save_django_messages([HumanMessage(content="Hello")], thread=thread)
    mock_bulk_create.assert_called_once()
    assert Message.objects.count() == 1
    assert Message.objects.first().message["data"]["content"] == "Hello"


@pytest.mark.django_db()
def test_django_messages_with_can_return_rows_from_bulk_insert_false():
    class MockFeatures(DatabaseFeatures):
        can_return_rows_from_bulk_insert = False

    mock_features = MockFeatures(connections[Message.objects.db])

    thread = baker.make(Thread, created_by=baker.make(User))
    with patch.object(
        Message.objects,
        "bulk_create",
        wraps=Message.objects.bulk_create,
    ) as mock_bulk_create:
        with patch.object(connection, "features", mock_features):
            save_django_messages([HumanMessage(content="Hello")], thread=thread)
    mock_bulk_create.assert_not_called()
    assert Message.objects.count() == 1
    assert Message.objects.first().message["data"]["content"] == "Hello"
