require_relative "../test/test_helper"
require_relative "and_gate"

describe Perceptron do
  it "should calc and gate" do
    assert_equal 0, Perceptron.and_gate(0, 0)
    assert_equal 0, Perceptron.and_gate(0, 1)
    assert_equal 0, Perceptron.and_gate(1, 0)
    assert_equal 1, Perceptron.and_gate(1, 1)
  end

  it "should calc and gate with bias" do
    assert_equal 0, Perceptron.and_gate_with_bias(0, 0)
    assert_equal 0, Perceptron.and_gate_with_bias(0, 1)
    assert_equal 0, Perceptron.and_gate_with_bias(1, 0)
    assert_equal 1, Perceptron.and_gate_with_bias(1, 1)
  end
end
