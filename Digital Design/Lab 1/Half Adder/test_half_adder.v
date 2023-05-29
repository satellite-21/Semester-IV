`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12.01.2022 23:17:59
// Design Name: 
// Module Name: test_half_adder
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module test_half_adder;
wire s, c;
reg a, b;

H1_half_adder N1(a, b, s, c);

initial
    begin
    a = 1'b0; b = 1'b0;
    #10 a = 1'b1;
    #10 b = 1'b1; a = 1'b0;
    #10 a = 1'b1;
    end

initial #50 $finish;
endmodule

