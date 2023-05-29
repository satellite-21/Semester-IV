`timescale 1ns / 1ps
module test_serial_adder;
reg [7:0] a, b;
reg clk;

wire [7:0] sum;
wire cout;

serial_adder sa(sum, cout, a, b, clk);

initial
    begin
        clk = 0;
        a = 8'b00110011; b=8'b01001100;
    end
 always #5 clk = ~clk;
 initial #100 $finish;
endmodule
