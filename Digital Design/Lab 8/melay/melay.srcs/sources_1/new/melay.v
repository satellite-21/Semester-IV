module melay_2s_complement(x, clk, rst, y, ps, ns);
input rst, clk, x;
output reg y;
output reg [1:0] ps, ns;
parameter s0 = 2'b00;
parameter s1 = 2'b01;

always @(posedge clk, posedge rst)
begin
	if(rst) ps<=s0;
	else ps<=ns;

end
always @(ps, x)
	case(ps)
		s0: if(x) ns<=s1;
		    else  ns<=s0;
		s1: if(x) ns<=s1;
	        else  ns<=s1;
		default : ns<=s0;
		endcase

always @(ps, x)
	case(ps)
		s0: y = x;
		s1: y=! x;
		endcase

endmodule

module MSL_using_LFSR(bit, load, clk, bits);
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