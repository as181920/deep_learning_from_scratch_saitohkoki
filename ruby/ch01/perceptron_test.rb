require_relative "../test/test_helper"
require_relative "perceptron"

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

  it "should calc nand gate" do
    assert_equal 1, Perceptron.nand_gate(0, 0)
    assert_equal 1, Perceptron.nand_gate(0, 1)
    assert_equal 1, Perceptron.nand_gate(1, 0)
    assert_equal 0, Perceptron.nand_gate(1, 1)
  end

  it "should calc or gate" do
    assert_equal 0, Perceptron.or_gate(0, 0)
    assert_equal 1, Perceptron.or_gate(0, 1)
    assert_equal 1, Perceptron.or_gate(1, 0)
    assert_equal 1, Perceptron.or_gate(1, 1)
  end

  it "should calc xor gate" do
    assert_equal 0, Perceptron.xor_gate(0, 0)
    assert_equal 1, Perceptron.xor_gate(0, 1)
    assert_equal 1, Perceptron.xor_gate(1, 0)
    assert_equal 0, Perceptron.xor_gate(1, 1)
  end
end
