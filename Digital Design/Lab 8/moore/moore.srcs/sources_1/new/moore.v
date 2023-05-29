`timescale 1ns / 1ps
module moore_2s_complement(x, clk, rst, y, state);
input rst, clk, x;
output reg y;
output reg [1:0] state;
parameter s0 = 2'b00;
parameter s1 = 2'b01;
parameter s2 = 2'b10;

always @(posedge clk, posedge rst)
begin
	if(rst) state<=s0;
	else
	begin 	
		case(state)
			s0: if(x) state <=s1;
			    else  state<=s0;
			
			s1: if(x) state <=s2;
			    else  state <=s1;


			s2: if(x) state <=s2;
			    else  state <=s1;

			default : state <= s0;
			endcase
		end
end

always @(state)
	case(state)
	s0: y = 0;
	s1: y = 1;
	s2: y = 0;
	endcase
endmodule

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