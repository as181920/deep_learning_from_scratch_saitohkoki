require_relative "../test/test_helper"
require_relative "gradient"

describe Gradient do
  before do
    @func1 = lambda do |x|
      ((x**2) * 0.01) + (x * 0.1)
    end

    @func2 = lambda do |x|
      (x**2).sum
      # (x[0]**2) + (x[1]**2)
    end
  end

  it "should implement numerical_diff" do
    expected = Torch.tensor(0.2, dtype: :float64)
    calculated = Gradient.numerical_diff(@func1, Torch.tensor(5.0, dtype: :float64))

    assert_operator (expected - calculated).abs, :<, 10**-4

    expected = Torch.tensor(0.3, dtype: :float64)
    calculated = Gradient.numerical_diff(@func1, Torch.tensor(10.0, dtype: :float64))

    assert_operator (expected - calculated), :<, 10**-4
  end

  it "should implement numerical gradient for" do
    expected = Torch.tensor([6.0, 8.0], dtype: :float64)
    calculated = Gradient.numerical_gradient(@func2, Torch.tensor([3.0, 4.0], dtype: :float64))

    expected.each_with_index do |value, index|
      assert_operator (value - calculated[index]).abs, :<, 10**-4
    end

    expected = Torch.tensor([0.0, 4.0], dtype: :float64)
    calculated = Gradient.numerical_gradient(@func2, Torch.tensor([0.0, 2.0], dtype: :float64))

    expected.each_with_index do |value, index|
      assert_operator (value - calculated[index]).abs, :<, 10**-4
    end

    expected = Torch.tensor([6.0, 0.0], dtype: :float64)
    calculated = Gradient.numerical_gradient(@func2, Torch.tensor([3.0, 0.0], dtype: :float64))

    expected.each_with_index do |value, index|
      assert_operator (value - calculated[index]).abs, :<, 10**-4
    end
  end

  it "should implement gradient_descent" do
    init_x = Torch.tensor([-3.0, 4.0], dtype: :float64)
    result = Gradient.gradient_descent(@func2, init_x, lr: 0.1)

    assert_operator result[0], :<, 10**-4
    assert_operator result[1], :<, 10**-4
  end
end
