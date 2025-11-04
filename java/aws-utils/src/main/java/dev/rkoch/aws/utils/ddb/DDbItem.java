package dev.rkoch.aws.utils.ddb;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import software.amazon.awssdk.services.dynamodb.model.AttributeValue;

public class DDbItem implements Map<String, AttributeValue> {

  private final Map<String, AttributeValue> map;

  @Override
  public String toString() {
    return "item=" + map;
  }

  public static DDbItem of(String key, int value) {
    DDbItem keyValueMap = new DDbItem();
    keyValueMap.put(key, AttributeValue.fromN(Integer.toString(value)));
    return keyValueMap;
  }

  public static DDbItem ofNumber(String key, String value) {
    DDbItem keyValueMap = new DDbItem();
    keyValueMap.put(key, AttributeValue.fromN(value));
    return keyValueMap;
  }

  public static DDbItem of(String key, String value) {
    DDbItem keyValueMap = new DDbItem();
    keyValueMap.put(key, AttributeValue.fromS(value));
    return keyValueMap;
  }

  public static DDbItem of(List<String> keys, List<String> values) {
    if (keys.size() != values.size()) {
      throw new IllegalArgumentException("keys and values count must be equal!");
    }
    DDbItem keyValueMap = new DDbItem();
    for (int i = 0; i < keys.size(); i++) {
      String value = values.get(i);
      keyValueMap.put(keys.get(i), AttributeValue.fromN(value));
    }
    return keyValueMap;
  }

  public static DDbItem of(final Map<String, AttributeValue> item, final String... keys) {
    DDbItem keyValueMap = new DDbItem();
    for (String key : keys) {
      keyValueMap.put(key, item.get(key));
    }
    return keyValueMap;
  }

  private DDbItem() {
    map = new LinkedHashMap<>();
  }

  @Override
  public int size() {
    return map.size();
  }

  @Override
  public boolean isEmpty() {
    return map.isEmpty();
  }

  @Override
  public boolean containsKey(Object key) {
    return map.containsKey(key);
  }

  @Override
  public boolean containsValue(Object value) {
    return map.containsValue(value);
  }

  @Override
  public AttributeValue get(Object key) {
    return map.get(key);
  }

  @Override
  public AttributeValue put(String key, AttributeValue value) {
    return map.put(key, value);
  }

  public AttributeValue put(String key, String value) {
    return map.put(key, AttributeValue.fromS(value));
  }

  public AttributeValue putNumber(String key, String value) {
    return map.put(key, AttributeValue.fromN(value));
  }

  public AttributeValue put(String key, int value) {
    return map.put(key, AttributeValue.fromN(Integer.toString(value)));
  }

  @Override
  public AttributeValue remove(Object key) {
    return map.remove(key);
  }

  @Override
  public void putAll(Map<? extends String, ? extends AttributeValue> m) {
    map.putAll(m);
  }

  @Override
  public void clear() {
    map.clear();
  }

  @Override
  public Set<String> keySet() {
    return map.keySet();
  }

  @Override
  public Collection<AttributeValue> values() {
    return map.values();
  }

  @Override
  public Set<Entry<String, AttributeValue>> entrySet() {
    return map.entrySet();
  }

  public String getUpdateExpression() {
    StringBuilder builder = new StringBuilder();
    builder.append("SET ");
    for (String key : map.keySet()) {
      builder.append("#");
      builder.append(replaceSpecialChars(key));
      builder.append("=:");
      builder.append(replaceSpecialChars(key));
      builder.append(",");
    }
    builder.deleteCharAt(builder.length() - 1);
    return builder.toString();
  }

  public String getUpdateIfNotExistsExpression() {
    StringBuilder builder = new StringBuilder();
    builder.append("SET ");
    for (String key : map.keySet()) {
      builder.append("#");
      builder.append(replaceSpecialChars(key));
      builder.append("=if_not_exists(#");
      builder.append(replaceSpecialChars(key));
      builder.append(",:");
      builder.append(replaceSpecialChars(key));
      builder.append(")");
      builder.append(",");
    }
    builder.deleteCharAt(builder.length() - 1);
    return builder.toString();
  }

  public String getUpdateExpression(final boolean updateExisting) {
    if (updateExisting) {
      return getUpdateExpression();
    } else {
      return getUpdateIfNotExistsExpression();
    }
  }

  public DDbItem toUpdateExpressionValues() {
    DDbItem keyValueMap = new DDbItem();
    for (Entry<String, AttributeValue> entry : this.entrySet()) {
      String key = entry.getKey();
      keyValueMap.put(":" + replaceSpecialChars(key), entry.getValue());
    }
    return keyValueMap;
  }

  private String replaceSpecialChars(final String key) {
    return key.replace('.', '_').replace('-', '_');
  }

  public DDbItem toUpdateExpressionValues(final String... keysToExclude) {
    Set<String> excludeSet = new HashSet<>(Arrays.asList(keysToExclude));
    DDbItem keyValueMap = new DDbItem();
    for (Entry<String, AttributeValue> entry : this.entrySet()) {
      String key = entry.getKey();
      if (!excludeSet.contains(key)) {
        keyValueMap.put(":" + key, entry.getValue());
      }
    }
    return keyValueMap;
  }

  public static DDbItem ofWithout(final Map<String, AttributeValue> item, final String... keysToExclude) {
    Set<String> excludeSet = new HashSet<>(Arrays.asList(keysToExclude));
    DDbItem keyValueMap = new DDbItem();
    for (Entry<String, AttributeValue> entry : item.entrySet()) {
      String key = entry.getKey();
      if (!excludeSet.contains(key)) {
        keyValueMap.put(key, entry.getValue());
      }
    }
    return keyValueMap;
  }

  public Map<String, String> getUpdateExpressionNames() {
    Map<String, String> updateExpressionNames = new HashMap<>();
    for (String key : map.keySet()) {
      updateExpressionNames.put("#" + replaceSpecialChars(key), key);
    }
    return updateExpressionNames;
  }

}
