package dev.rkoch.aws.s3.parquet;

import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.parquet.schema.MessageType;
import blue.strategic.parquet.Dehydrator;
import blue.strategic.parquet.Hydrator;
import blue.strategic.parquet.HydratorSupplier;
import blue.strategic.parquet.ParquetReader;
import blue.strategic.parquet.ParquetWriter;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.http.urlconnection.UrlConnectionHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

public class S3Parquet {

  private final S3Client s3Client;

  S3Parquet() {
    this(S3Client.builder().httpClientBuilder(UrlConnectionHttpClient.builder()).region(Region.EU_WEST_1).build());
  }

  public S3Parquet(S3Client s3Client) {
    this.s3Client = s3Client;
  }

  public <T extends ParquetRecord<T>> List<T> read(final String bucket, final String key, final Class<T> hydratorClass) throws Exception {
    Path path = Files.createTempFile(null, null);
    try (ResponseInputStream<GetObjectResponse> inputStream = s3Client.getObject(GetObjectRequest.builder().bucket(bucket).key(key).build());
        OutputStream outputStream = Files.newOutputStream(path)) {
      inputStream.transferTo(outputStream);
    }
    Hydrator<T, T> hydrator = hydratorClass.getConstructor().newInstance().getHydrator();
    return ParquetReader.streamContent(path.toFile(), HydratorSupplier.constantly(hydrator)).toList();
  }

  public <T extends ParquetRecord<T>> void write(final String bucket, final String key, final List<T> records) throws Exception {
    if (!records.isEmpty()) {
      Path tempFile = Files.createTempFile(null, null);
      T first = records.getFirst();
      /**
       * https://github.com/strategicblue/parquet-floor/blob/master/src/test/java/blue/strategic/parquet/ParquetReadWriteTest.java
       */
      MessageType schema = first.getSchema();
      Dehydrator<T> dehydrator = first.getDehydrator();
      try (ParquetWriter<T> writer = ParquetWriter.writeFile(schema, tempFile.toFile(), dehydrator)) {
        for (T record : records) {
          writer.write(record);
        }
      }
      s3Client.putObject(PutObjectRequest.builder().bucket(bucket).key(key).build(), RequestBody.fromFile(tempFile));
    }
  }

}
