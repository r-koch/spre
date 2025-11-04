package dev.rkoch.aws.utils.lambda;

import java.io.IOException;
import com.amazonaws.services.lambda.runtime.LambdaLogger;

public class SystemOutLambdaLogger implements LambdaLogger {

  @Override
  public void log(String message) {
    System.out.println(message);
  }

  @Override
  public void log(byte[] message) {
    try {
      System.out.write(message);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

  }

}
