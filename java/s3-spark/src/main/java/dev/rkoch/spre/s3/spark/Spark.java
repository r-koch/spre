package dev.rkoch.spre.s3.spark;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import org.apache.iceberg.catalog.Namespace;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.spark.SparkCatalog;
import org.apache.iceberg.spark.actions.SparkActions;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.catalog.Identifier;
import software.amazon.awssdk.regions.Region;

public class Spark<T extends DatedItem> {

  static {
    System.setProperty("aws.region", Region.EU_WEST_1.toString());
  }

  private final Class<T> type;
  private final String bucket;
  private final String catalog;
  private final String namespace;
  private final String table;
  private final String sparkTable;

  public Spark(Class<T> type, String bucket, String catalog, String namespace, String table) {
    this.type = type;
    this.bucket = bucket;
    this.catalog = catalog;
    this.namespace = namespace;
    this.table = table;
    sparkTable = String.join(".", catalog, namespace, table);
  }

  public static void main(String[] args) {
    List<NewsItem> data = List.of(NewsItem.of(LocalDate.parse("2025-09-21"), "1", "Markets rally on Fed pause", "Bloomberg"),
        NewsItem.of(LocalDate.parse("2025-09-21"), "2", "Oil prices hit new high", "Reuters"));

    Spark<NewsItem> spark = new Spark<NewsItem>(NewsItem.class, "dev-rkoch-spre-test", "glue", "blub", "news");
    spark.insert(data);
    spark.insert(data);
    spark.show();
    spark.deduplicate();
    spark.show();
  }

  private SparkSession getSparkSession() {
    return SparkSession.builder() //
        .master("local[*]") //
        .config("spark.sql.catalog." + catalog, "org.apache.iceberg.spark.SparkCatalog") //
        .config("spark.sql.catalog." + catalog + ".io-impl", "org.apache.iceberg.aws.s3.S3FileIO") //
        .config("spark.sql.catalog." + catalog + ".type", "glue") //
        .config("spark.sql.catalog." + catalog + ".warehouse", "s3://" + bucket + "/iceberg/") //
        .getOrCreate();
  }

  public void show() {
    try (SparkSession spark = getSparkSession()) {

      SparkCatalog sparkCatalog = (SparkCatalog) spark.sessionState().catalogManager().catalog(catalog);

      String[] ns = Namespace.of(namespace).levels();
      if (sparkCatalog.namespaceExists(ns) && sparkCatalog.tableExists(Identifier.of(ns, table))) {
        spark.read().table(sparkTable).show();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void insert(final List<T> data) {
    try (SparkSession spark = getSparkSession()) {

      SparkCatalog sparkCatalog = (SparkCatalog) spark.sessionState().catalogManager().catalog(catalog);

      String[] ns = Namespace.of(namespace).levels();
      if (!sparkCatalog.namespaceExists(ns)) {
        sparkCatalog.createNamespace(ns, Map.of());
      }

      Dataset<Row> df = spark.createDataFrame(data, type);

      if (sparkCatalog.tableExists(Identifier.of(ns, table))) {
        df.writeTo(sparkTable).append();
      } else {
        df.writeTo(sparkTable).partitionedBy(df.col("localDate")).createOrReplace();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void deduplicate() {
    try (SparkSession spark = getSparkSession()) {

      SparkCatalog sparkCatalog = (SparkCatalog) spark.sessionState().catalogManager().catalog(catalog);

      String[] ns = Namespace.of(namespace).levels();
      if (sparkCatalog.namespaceExists(ns) && sparkCatalog.tableExists(Identifier.of(ns, table))) {
        Dataset<Row> existing = spark.read().table(sparkTable);
        Dataset<Row> deduped = existing.dropDuplicates("id", "localDate");
        deduped.writeTo(sparkTable).overwritePartitions();

        SparkActions.get(spark).rewriteDataFiles(sparkCatalog.icebergCatalog().loadTable(TableIdentifier.of(namespace, table))).execute();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

}
