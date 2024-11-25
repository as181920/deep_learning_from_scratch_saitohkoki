require_relative "../test/test_helper"
require_relative "../common/global"
require_relative "mul_layer"

describe MulLayer do
  it "forward and backward for multiply layer" do
    apple = 100
    apple_num = 2
    tax = 1.1

    # set layer
    mul_num_layer = MulLayer.new
    mul_tax_layer = MulLayer.new

    # forward
    apple_amount = mul_num_layer.forward(apple, apple_num)
    taxed_amount = mul_tax_layer.forward(apple_amount, tax)

    assert_in_delta 220, taxed_amount, 10**-6

    # backward
    dprice = 1
    dapple_amount, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_num_layer.backward(dapple_amount)

    assert_in_delta 2.2, dapple, 10**-6
    assert_in_delta 110, dapple_num, 10**-6
    assert_in_delta 200, dtax, 10**-6
  end
end
