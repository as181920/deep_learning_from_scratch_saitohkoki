require_relative "../test/test_helper"
require_relative "and_gate"

describe Perceptron do
  it "should calc and gate" do
    assert_equal 0, Perceptron.and_gate(0.1, 0.2)
  end
end
