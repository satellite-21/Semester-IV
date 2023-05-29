`timescale 1ns / 1ps
module test_bench();
wire [31:0]s;
wire cout;
reg [31:0]a;
reg [31:0]b;
CLA_Adder cla32(a,b,s,cout);
initial
    begin
        a=32'b00000000000000000000000000000000;
        b=32'b00000000000000000000000000000000;
    end
    always #100 b = b+2'b10;
    always #100 a = a+2'b10;
initial #6000 $finish;
endmodule
