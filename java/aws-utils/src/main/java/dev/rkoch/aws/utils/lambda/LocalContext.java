package dev.rkoch.aws.utils.lambda;

import com.amazonaws.services.lambda.runtime.ClientContext;
import com.amazonaws.services.lambda.runtime.CognitoIdentity;
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.LambdaLogger;

public class LocalContext implements Context {

  @Override
  public String getAwsRequestId() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String getLogGroupName() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String getLogStreamName() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String getFunctionName() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String getFunctionVersion() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String getInvokedFunctionArn() {
    throw new UnsupportedOperationException();
  }

  @Override
  public CognitoIdentity getIdentity() {
    throw new UnsupportedOperationException();
  }

  @Override
  public ClientContext getClientContext() {
    throw new UnsupportedOperationException();
  }

  @Override
  public int getRemainingTimeInMillis() {
    return Integer.MAX_VALUE;
  }

  @Override
  public int getMemoryLimitInMB() {
    throw new UnsupportedOperationException();
  }

  @Override
  public LambdaLogger getLogger() {
    return new SystemOutLambdaLogger();
  }

}
