`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12.01.2022 23:19:29
// Design Name: 
// Module Name: H1_half_adder
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

module H1_half_adder(a, b, s, c);

input a, b;
output s, c;

assign s = a^b;
assign c = a&b;


endmodule
