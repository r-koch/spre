package dev.rkoch.aws.utils.ddb;

import software.amazon.awssdk.services.dynamodb.model.DeleteItemRequest;
import software.amazon.awssdk.services.dynamodb.model.GetItemRequest;
import software.amazon.awssdk.services.dynamodb.model.GetItemResponse;

public interface DefaultVariable {

  String name();

  String getTable();

  default String get() {
    GetItemResponse item = DDb.client().getItem(GetItemRequest.builder().tableName(getTable()).key(DDbItem.of("pk_name", name())).build());
    return item.item().get("value").s();
  }

  default String set(final String value) {
    DDbItem keyItem = DDbItem.of("pk_name", name());
    DDbItem valueItem = DDbItem.of("value", value);
    DDb.insert(getTable(), keyItem, valueItem, true);
    return value;
  }

  default void clear() {
    DDb.client().deleteItem(DeleteItemRequest.builder().tableName(getTable()).key(DDbItem.of("pk_name", name())).build());
  }

}
