package dev.rkoch.spre.aws.utils.ddb;

import java.util.Map;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.ReturnValue;
import software.amazon.awssdk.services.dynamodb.model.UpdateItemRequest;
import software.amazon.awssdk.services.dynamodb.model.UpdateItemResponse;

public enum DDb {

  INSTANCE;

  private static final DynamoDbClient DB_CLIENT = DynamoDbClient.builder().region(Region.EU_WEST_1).build();

  public static DynamoDbClient client() {
    return DB_CLIENT;
  }

  /**
   * @return true if insert; false if update
   */
  public static boolean insert(final String tableName, final DDbItem item, final String partitionKey) {
    return insert(tableName, item, partitionKey, false);
  }

  /**
   * @return true if insert; false if update
   */
  public static boolean insert(final String tableName, final DDbItem item, final String partitionKey, final boolean overwrite) {
    DDbItem values = DDbItem.ofWithout(item, partitionKey);
    String updateExpression = values.getUpdateExpression(overwrite);
    Map<String, String> updateExpressionNames = values.getUpdateExpressionNames();
    DDbItem updateExpressionValues = values.toUpdateExpressionValues();
    UpdateItemResponse response = DB_CLIENT.updateItem(UpdateItemRequest.builder().tableName(tableName) //
        .key(DDbItem.ofNumber(partitionKey, item.get(partitionKey).n())) //
        .updateExpression(updateExpression) //
        .expressionAttributeNames(updateExpressionNames) //
        .expressionAttributeValues(updateExpressionValues) //
        .returnValues(ReturnValue.UPDATED_OLD) //
        .build());
    return response.attributes().isEmpty();
  }

  /**
   * @return true if insert; false if update
   */
  public static boolean insert(final String tableName, final DDbItem key, final DDbItem values, final boolean overwrite) {
    String updateExpression = values.getUpdateExpression(overwrite);
    Map<String, String> updateExpressionNames = values.getUpdateExpressionNames();
    DDbItem updateExpressionValues = values.toUpdateExpressionValues();
    UpdateItemResponse response = DB_CLIENT.updateItem(UpdateItemRequest.builder().tableName(tableName) //
        .key(key) //
        .updateExpression(updateExpression) //
        .expressionAttributeNames(updateExpressionNames) //
        .expressionAttributeValues(updateExpressionValues) //
        .returnValues(ReturnValue.UPDATED_OLD) //
        .build());
    return response.attributes().isEmpty();
  }

}
