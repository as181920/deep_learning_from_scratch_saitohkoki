require_relative "../test/test_helper"
require_relative "../common/global"
require_relative "add_layer"
require_relative "mul_layer"

describe AddLayer do
  it "forward and backward for multiply layer" do
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    # layer
    mul_apple_layer = MulLayer.new
    mul_orange_layer = MulLayer.new
    add_apple_orange_layer = AddLayer.new
    mul_tax_layer = MulLayer.new

    # forward
    apple_amount = mul_apple_layer.forward(apple, apple_num)
    orange_amount = mul_orange_layer.forward(orange, orange_num)
    all_amount = add_apple_orange_layer.forward(apple_amount, orange_amount)
    taxed_amount = mul_tax_layer.forward(all_amount, tax)

    assert_in_delta 715, taxed_amount, 10**-6

    # backward
    dprice = 1
    dall_amount, dtax = mul_tax_layer.backward(dprice)
    dapple_amount, dorange_amount = add_apple_orange_layer.backward(dall_amount)
    dorange, dorange_num = mul_orange_layer.backward(dorange_amount)
    dapple, dapple_num = mul_apple_layer.backward(dapple_amount)

    assert_in_delta 650, dtax, 10**-6
    assert_in_delta 165, dorange_num, 10**-6
    assert_in_delta 3.3, dorange, 10**-6
    assert_in_delta 110, dapple_num, 10**-6
    assert_in_delta 2.2, dapple, 10**-6
  end
end
