`timescale 1ns / 1ps
module MSL_using_LSFR(bit, load, clk, bits);
parameter n = 8;
input load, clk;
output reg [n-1:0]bits;
output reg bit;
reg store;
always @(posedge clk, posedge load)
if(load)
	bits = 8'b10101100;
else
begin
bit  = bits[0];
store = bits[0]^bits[2]^bits[3]^bits[4];
bits = bits>>1;
bits[7] = store;
end
endmodule
