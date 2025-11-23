package dev.rkoch.spre.stock.collector.exception;

public class SymbolNotFoundException extends Exception {

  private static final long serialVersionUID = 1L;

  private String symbol;

  @Override
  public String getMessage() {
    return "%s not found".formatted(symbol);
  }

  public void setSymbol(String symbol) {
    this.symbol = symbol;
  }

}
