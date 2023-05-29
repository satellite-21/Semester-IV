`timescale 1ns / 1ps
module testBench;
wire [31:0]s;
wire cout;
reg [31:0]a;
reg [31:0]b;
carrySelectAdder32bit oi(a,b,s,cout);
initial
    begin
    a=32'b00000000000000000000000000000000;b=32'b00000000000000000000000000000000;
    #100 a=32'b00000000000000000000000000000010;b=32'b00000000000000000000000000010001;
    #100 a=32'b00000000000000000000000000001000;b=32'b00000000000000000000000000011000;
    #100 a=32'b00000000000000000000000000000100;b=32'b00000000000000000000000000110000;
    end
initial #500 $finish;
endmodule