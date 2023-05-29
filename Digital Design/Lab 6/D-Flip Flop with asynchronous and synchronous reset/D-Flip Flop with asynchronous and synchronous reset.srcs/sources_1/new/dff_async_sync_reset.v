`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04.03.2022 17:26:57
// Design Name: 
// Module Name: dff_async_sync_reset
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

module dff_async_sync_reset(d1, d2, reset, clk,o1,o2);
input d1, d2, reset, clk;
output reg o1, o2;

always@(posedge clk)
	if(reset)
		o1=0;
	else
		o1=d1;

always@(posedge clk or posedge reset)
	if(reset)
		o2 = 0;
	else
		o2 = d2;

endmodule
