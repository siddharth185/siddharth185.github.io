

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
%matplotlib inline
```


```python
df = pd.read_csv("parliament.csv")
```


```python
plt.rcParams["figure.figsize"] = [10,8]
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>STATE</th>
      <th>PC</th>
      <th>NAME</th>
      <th>SEX</th>
      <th>PARTY</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>VOTES</th>
      <th>ELECTORS</th>
      <th>#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>JAWALA PRASHAD</td>
      <td>NaN</td>
      <td>INC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46679</td>
      <td>162327.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>CHAND KARAN</td>
      <td>NaN</td>
      <td>BJS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28990</td>
      <td>162327.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>DINO MAL</td>
      <td>NaN</td>
      <td>PURP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10778</td>
      <td>162327.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>BAJORIA BADRIDAS</td>
      <td>NaN</td>
      <td>IND</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6153</td>
      <td>162327.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>RANG RAJ MEHTA</td>
      <td>NaN</td>
      <td>IND</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4565</td>
      <td>162327.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(include = "all")
```

    /usr/local/lib/python2.7/dist-packages/numpy/lib/function_base.py:3823: RuntimeWarning: Invalid value encountered in percentile
      RuntimeWarning)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>STATE</th>
      <th>PC</th>
      <th>NAME</th>
      <th>SEX</th>
      <th>PARTY</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>VOTES</th>
      <th>ELECTORS</th>
      <th>#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>74930.000000</td>
      <td>74930</td>
      <td>74930</td>
      <td>74930</td>
      <td>71462</td>
      <td>74930</td>
      <td>13505.000000</td>
      <td>13505</td>
      <td>74930.000000</td>
      <td>6.949500e+04</td>
      <td>74930.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>50</td>
      <td>973</td>
      <td>58067</td>
      <td>2</td>
      <td>881</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>UTTAR PRADESH</td>
      <td>NALGONDA</td>
      <td>OM PRAKASH</td>
      <td>M</td>
      <td>IND</td>
      <td>NaN</td>
      <td>GEN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>15582</td>
      <td>575</td>
      <td>80</td>
      <td>68349</td>
      <td>41469</td>
      <td>NaN</td>
      <td>9246</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1989.949473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.774972</td>
      <td>NaN</td>
      <td>49700.452969</td>
      <td>9.573475e+05</td>
      <td>11.962859</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.981721</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.923001</td>
      <td>NaN</td>
      <td>94144.950189</td>
      <td>3.516189e+05</td>
      <td>30.865537</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1951.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1984.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>940.000000</td>
      <td>NaN</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1991.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3535.000000</td>
      <td>NaN</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1998.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45776.750000</td>
      <td>NaN</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2009.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99.000000</td>
      <td>NaN</td>
      <td>855543.000000</td>
      <td>3.103525e+06</td>
      <td>480.000000</td>
    </tr>
  </tbody>
</table>
</div>



We notice that there are 50 states and 973 unique constituencies.
From 1951 to 2009 constiuencies were changed many times and hence the large number.
Lets look at the unique values.



```python
print df.CATEGORY.unique()
print "\n"
print df.STATE.unique()
print "\n"
print df.PARTY.unique()
print "\n"

```

    ['GEN' 'ST' 'SC']
    
    
    ['AJMER' 'ASSAM' 'BHOPAL' 'BIHAR' 'BILASPUR' 'BOMBAY' 'COORG'
     'HIMACHAL PRADESH' 'HYDERABAD' 'KUTCH' 'MADHYA BHARAT' 'MADHYA PRADESH'
     'MADRAS' 'MANIPUR' 'MYSORE' 'NCT OF DELHI' 'ORISSA'
     'PATIALA AND EAST PUNJAB STATES UNION' 'PUNJAB' 'RAJASTHAN' 'SAURASHTRA'
     'TRAVANCORE COCHIN' 'TRIPURA' 'UTTAR PRADESH' 'VINDHYA PRADESH'
     'WEST BENGAL' 'ANDHRA PRADESH' 'KERALA' 'GUJARAT' 'MAHARASHTRA'
     'A&N ISLANDS' 'CHANDIGARH' 'D&N HAVELI' 'GOA, DAMAN & DIU' 'HARYANA'
     'JAMMU & KASHMIR' 'LAKSHADWEEP' 'NAGALAND' 'PUDUCHERRY' 'TAMIL NADU'
     'ARUNACHAL PRADESH' 'KARNATAKA' 'MEGHALAYA' 'MIZORAM' 'SIKKIM'
     'DAMAN & DIU' 'GOA' 'CHHATTISGARH' 'JHARKHAND' 'UTTARAKHAND']
    
    
    ['INC' 'BJS' 'PURP' 'IND' 'CPI' 'RRP' 'KJD' 'HPP' 'SP' 'KMPP' 'TS' 'APP'
     'HMS' 'KMM' 'JHP' 'FBL(MG)' 'UKS' 'CNSPJP' 'LSS' 'SCF' 'KLP' 'KKP' 'NAT'
     'PWP' 'PDF' 'HSPP' 'FBL(RG)' 'SKP' 'REP' 'TNT' 'CWL' 'JUSP' 'ML' 'GSS'
     'PP' 'AMN' 'HR' 'KNA' 'GP' 'SAD' 'RCPI' 'ZP' 'DCL' 'KJSP' 'SKS' 'RPP' 'CP'
     'KSP' 'TTC' 'TP' 'RSP' 'RSP(UP)' 'UPPP' 'BPI' 'PSP' 'FBM' 'SWA' 'JS' 'HLC'
     'SOC' 'RCP' 'JP' 'NJP' 'FB' 'DMK' 'SL' 'TNP' 'WT' 'AD' 'HLS' 'EIT' 'GL'
     'CPM' 'RPI' 'SSP' 'AHL' 'JKD' 'UGS' 'UGF' 'JKN' 'DNC' 'KEC' 'MUL' 'JAC'
     'NNO' 'PFR' 'ADS' 'ADM' 'FBL' 'BAC' 'TPS' 'NCO' 'RPK' 'MLP' 'TEC' 'BCM'
     'SUC' 'RCI' 'BKD' 'HSD' 'PHJ' 'JKP' 'ISP' 'JAP' 'SHD' 'BRP' 'PBI' 'RSM'
     'VHP' 'MAG' 'LRP' 'SHS' 'RPG' 'MRP' 'UFN' 'UTC' 'RPA' 'KMP' 'MUM' 'IGL'
     'BLD' 'SSD' 'KCP' 'MLO' 'UDF' 'ADK' 'ILP' 'JMD' 'TUS' 'INC(I)' 'JNP'
     'JNP(S)' 'INC(U)' 'BSP' 'PPA' 'IML' 'PPC' 'SJP' 'SCR' 'SPC' 'LKD' 'TDP'
     'BJP' 'ICS' 'ICJ' 'JMM' 'DDP' 'JPP' 'KCJ' 'NND' 'GKC' 'TNC' 'PTC'
     'ICS(SCS)' 'JNP (JP)' 'MCPI' 'MIM' 'LKP' 'LKD (B)' 'JD' 'SOP(L)' 'DLP'
     'VJS' 'IPF' 'BKS' 'BKUS' 'PVP' 'M-COR' 'HJD' 'AMB' 'BLKD' 'LMS' 'DMM'
     'SAD(M)' 'GJP' 'PDL' 'KRS' 'SVP' 'KGP' 'DKP' 'KCM' 'NRP' 'GPI' 'HPI' 'HJP'
     'VBP' 'SHP' 'MPC' 'MNF' 'DP' 'NPC' 'PPI' 'LPI' 'HSS' 'ICJ(TG)' 'PDI' 'LMD'
     'RPI(GG)' 'OPF' 'PMK' 'TMM' 'SAD(B)' 'IGC' 'PNF' 'BJMD' 'PPP' 'PKD'
     'CPI(ML)' 'SRP' 'RIS' 'TRM' 'MPP' 'SLL' 'PWA' 'IJP' 'BMC' 'KDC' 'UPRP'
     'SOP' 'UKD' 'LTP' 'INC(O)' 'GNLF' 'GOL(BG)' 'WBS(BM)' 'LAB(B)' 'NBP' 'MSD'
     'BDLP' 'ADC' 'AGP' 'UMF' 'NAGP' 'URC' 'AJD' 'SLP' 'JD(S)' 'PTCA' 'AJP'
     'SKD' 'DBM' 'ABSP' 'JMS' 'AZP' 'RRP(S)' 'JF' 'LAJM' 'KMS' 'GLP' 'YVP'
     'SLI' 'HSH' 'JD(G)' 'AHF' 'SCP' 'SMP' 'HVP' 'JKMP' 'SJJP' 'RKD' 'SUP'
     'KNP' 'CKN' 'BBP' 'KDP' 'GNP' 'DPI' 'BDP' 'RPPI' 'VPP' 'DPP' 'SRS'
     'RPI(KM)' 'RSD' 'NP' 'NPP' 'DBP' 'RPI(A)' 'BLMD' 'RUD' 'DND' 'PRC'
     'LPI(P)' 'BSNP' 'AUM' 'JEM' 'MUB' 'LHM' 'JC' 'OVP' 'PMM' 'IDP' 'MHA'
     'MGMK' 'TMK' 'AMI' 'GMK' 'SOP(RP)' 'MDL' 'ABGP' 'ALD' 'SHS(R)' 'HKSP'
     'JJP' 'WPI' 'NTRTDP(LP)' 'RPI(KH)' 'MBT' 'AIIC(T)' 'SHSP' 'SAP' 'MCPI(S)'
     'AIMIM' 'PHK' 'SYP' 'CPI(ML)(L)' 'LP' 'IUML' 'NIP' 'ABRRP(P)' 'ASDC'
     'UMFA' 'UTNLF' 'RCPI(R)' 'RMEP' 'VJP' 'ABDBM' 'BLPY' 'ABJVP' 'PSSS' 'RAM'
     'MCO' 'JMM(M)' 'ABJS' 'JKPP' 'JKP(N)' 'ABLTASJM' 'BSP(A)' 'UGDP' 'GAVP'
     'RSRP' 'BHJS' 'MSS' 'GJJP' 'BMSM' 'JHM' 'ABDUP' 'PPNMS' 'JSTP' 'BLP'
     'BKD(J)' 'RKP' 'ARS' 'JPS' 'RNP' 'BNJP' 'KCVP' 'KP' 'UIDC' 'PDP' 'INL'
     'KEC(M)' 'KSM' 'RSPP' 'BMRD' 'SDP(MP)' 'LD' 'MPVC' 'MKVP' 'CMM' 'ABHM'
     'SVSP' 'SLP(L)' 'RPI(D)' 'GGP' 'ABBNS' 'SJP(M)' 'PWPI' 'NVAS' 'BBM' 'SBHP'
     'BRC' 'COP' 'PDLI' 'MPKP' 'FPM' 'ESP' 'ABMSKP' 'ABDD' 'ABSR' 'LPI(V)'
     'AIAHMJKP' 'SABJAN' 'BSP(R)' 'BAZP' 'BLTMD' 'BCVD' 'BSVP' 'ISC' 'MSP'
     'BPM' 'BLTP' 'MB' 'HNP' 'BRD' 'RSDP' 'BPF' 'BOP' 'HJKP' 'MDMK' 'ABMSD'
     'LHP' 'PVP(P)' 'SWD' 'ABRAHP' 'SDF' 'TMC(M)' 'MELPHC' 'ATMK' 'RPI(S)'
     'ADMK' 'MMS' 'AIDPF' 'TNHVYK' 'TUJS' 'BEP' 'JKC' 'HKMP' 'BSK' 'AKD' 'ABBP'
     'BJKP' 'IBSP' 'NDPF' 'ABLTP' 'ABRRP' 'ABRS' 'PMSP' 'BRM' 'SLP(R)' 'FOS'
     'BSSM' 'BJTP' 'ABKMM' 'RSSD' 'GSP' 'BHJP' 'JSWP' 'KVMP' 'ABJND' 'KRD'
     'EKD(UP)' 'PSS' 'RBNNS' 'PBRML' 'FB(S)' 'IDPP' 'ANP' 'RJD' 'AJBP' 'LS'
     'SJP(R)' 'AIRJP' 'JTP' 'TGPP' 'AC' 'URMCA' 'BJC' 'SWJP' 'BMF' 'AIIC(S)'
     'AIMF' 'BKKGP' 'BLLP' 'LSWP' 'HLD(R)' 'ABHS' 'NPAP' 'HVC' 'JKAL' 'KTVP'
     'KRRS' 'RJDP' 'MP' 'THPI' 'BAJP' 'SSJP' 'AIRKC' 'MRC' 'ABRSSD' 'NLP'
     'MSCP' 'UDP' 'HPDP' 'GNC' 'RAS' 'AGRJP' 'UCP' 'RMMP' 'ABGLP' 'RJC' 'RRCD'
     'JSMP' 'ABJC' 'ABGMKP' 'BRVP' 'HM' 'BJJP' 'BJD' 'OCG' 'BKKP' 'AIGC' 'PPVP'
     'ABRC(D)' 'RJVP' 'GIP' 'PRISM' 'UCPI' 'MADMK' 'PT' 'TLJ' 'HIP' 'KVP' 'ABP'
     'AIMLF' 'ABAS' 'IRF' 'BVP' 'UKKD' 'WBTC' 'TRMRPPI' 'NCP' 'ATDP' 'PPOI'
     'UBNLF' 'BHJC' 'JD(U)' 'KVSB' 'RLD' 'CVP' 'BJC(R)' 'JMM(U)' 'SPSP' 'BHJVP'
     'BPSP' 'GVP' 'GYVP' 'INLD' 'AJKPPF' 'JKNPP' 'SPI' 'BRPP' 'PSJP' 'RDNBP'
     'CSP' 'NMP' 'GRC' 'SBP' 'PDM' 'MDF' 'EU' 'NCY' 'ABMAD' 'BD' 'BEP(R)' 'AP'
     'SHSAD' 'BMP(AI)' 'DBSM' 'MKSD' 'TDK' 'DTMK' 'LJSPI' 'NOC' 'INGP' 'TRC'
     'TNPWP' 'CMK' 'THMM' 'AITC' 'ASP' 'BBMKD' 'BNJS' 'HDVP' 'ABLTC' 'JSAP'
     'BND' 'BSD' 'LJP' 'TRS' 'PRBP' 'ANC' 'MRS' 'TNGP' 'KVSP' 'LSD' 'PTSS'
     'RSMD' 'SBSP' 'BED' 'SPVD' 'JDP' 'KSVP' 'RLSM' 'JHSP' 'BSJM' 'YGP'
     'BPSGKD' 'BNP' 'JMP' 'ES' 'LP(S)' 'RGD' 'FCI' 'RVP' 'NSSP' 'AIFB' 'RSKP'
     'RSBP' 'JCP' 'JKPDP' 'SDP' 'AJSU' 'NPF' 'KNDP' 'USYP' 'RSPS' 'SLAP' 'IFDP'
     'AKMDMP' 'SMSP' 'RSGP' 'ARP' 'PRP' 'PRCP' 'RSNP' 'SVRP' 'NSTP' 'KKJHS'
     'MRRC' 'VRP' 'VJC' 'HEAP' 'AB' 'PMP' 'BPTP' 'NBNP' 'RVNP' 'BGTD' 'MJM'
     'RJAP' 'BMVP' 'DBSP' 'MB(S)P' 'PBLP' 'LBP' 'LPSP' 'ABCD(A)' 'SHRP' 'JJ'
     'YSP' 'RHD' 'RKSP' 'MNVP' 'BKRP' 'LCP' 'NSP' 'PHSP' 'MC' 'BNRP' 'NMNP'
     'SBS' 'BJVP' 'BSDP' 'VP' 'JSP' 'RPD' 'JVP' 'BKLJP' 'PDS' 'JUM' 'PRAP'
     'TPPP' 'PBHP' 'RDMP' 'LSP' 'MANP' 'GRIP' 'RDHP' 'BJSH' 'RRS' 'BHSASP'
     'BPD' 'BSP(AP)' 'BCUF' 'UWF' 'RPC(S)' 'BSSP' 'LB' 'AUDF' 'RWS' 'BOPF'
     'BVM' 'RKJP' 'RJJM' 'RSWD' 'JKM' 'LTSD' 'BSKP' 'JVM' 'JGP' 'BHJAP' 'BJJD'
     'BSP(K)' 'LM' 'STPI' 'AJSP' 'BJKVP' 'EKSP' 'BJKD' 'BUDM' 'BLPGL' 'SJTP'
     'SUSP' 'BHPD' 'CGVP' 'GMS' 'RGOP' 'JCGP' 'ABSSP' 'SSBD' 'SGF' 'MJP' 'BNJD'
     'NLHP' 'SVPP' 'RSP(S)' 'VHS' 'NSCP' 'RPIE' 'ADSP' 'HJCBL' 'SMBHP' 'BHBP'
     'AIFB(S)' 'RND' 'JJJKMC' 'AIRP' 'JUP' 'RASAP' 'RASJP' 'NELU' 'VAJP' 'BHC'
     'RRD' 'JKANC' 'RNSP' 'JPC' 'BCDP' 'DGPP' 'BSKRP' 'JVD' 'AJSUP' 'BSA'
     'RASD' 'JHJAM' 'LJVM' 'MMM' 'BSSPA' 'JHJM' 'KTMK' 'BPJP' 'AIJMK' 'NDEP'
     'ICSP' 'BSC' 'BHPP' 'VCK' 'BMM' 'IVD' 'PRSP' 'BJBP' 'LKSP' 'ABHKP' 'AIC'
     'DESEP' 'RP(K)' 'KM' 'MNS' 'STBP' 'ABMP' 'LKSGM' 'SWP' 'JSS' 'PRPI' 'PPIS'
     'BREM' 'BVA' 'LVKP' 'PG' 'RBCP' 'PDA' 'HSPDP' 'MDP' 'ACNC' 'BPC' 'MBP'
     'BVVP' 'YFE' 'AWD' 'AIBS' 'LKJP' 'JBP' 'RALP' 'KS' 'SAMO' 'JHKP' 'KOKD'
     'OMM' 'DMDK' 'DCP' 'ARWP' 'AIDWC' 'ABJP' 'PLP' 'RMGLMP' 'RBD' 'BCP' 'RDSD'
     'SJEP' 'SGPP' 'PNK' 'AIVP' 'ADSMK' 'MAMAK' 'MMKA' 'PKMK' 'DPK' 'KNMK'
     'UMK' 'CDF' 'NMK' 'JBSP' 'RAJUP' 'MADP' 'RMSP' 'PECP' 'MAP' 'MD' 'MMUP'
     'BDBP' 'BUM' 'PDFO' 'JM' 'MKUP' 'BSRD' 'RJPK' 'UNLP' 'RTKP' 'RLP' 'PRBD'
     'BJBCD' 'RSUPRP' 'RYS' 'BAP' 'BSKPB' 'LSVP' 'RWSP' 'IPP' 'RALOP' 'SWPI'
     'BGD' 'JANS' 'RAD' 'MOP' 'APRD' 'NYP' 'MKD' 'RJSD' 'VVS' 'IPFB']
    
    


A lot of Missing Values in Category and Age. Only 18% values are present.

By further analysis it seems that ages and Category were only recorded from 2004 onwards and all values before that were missing.

We can replace Age by the mean since the SD is only 11years it is a good representative of the Age.

From Wikipedia - 

Out of 543 constituencies represented in the Lok Sabha, the lower house India's parliament, 84 (15.47%) are reserved for SC/Dalits and 47 (8.66%) are reserved for ST.

To fill the missing values in Category, I randomly replace the missing values randomly with SC, ST and GEN in the above mentioned ratio.



```python
df['AGE'].fillna((df['AGE'].mean()), inplace=True)

df['CATEGORY'].fillna(pd.Series(np.random.choice(['SC', 'ST', 'GEN'], 
                                                      p=[0.1547, 0.0866, 0.7587], size=len(df))),inplace=True)
df['SEX'].fillna("M",inplace=True)

df['ELECTORS'].fillna((df['ELECTORS'].mean()), inplace = True)
```


```python
df.describe(include = "all")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>STATE</th>
      <th>PC</th>
      <th>NAME</th>
      <th>SEX</th>
      <th>PARTY</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>VOTES</th>
      <th>ELECTORS</th>
      <th>#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>74930.000000</td>
      <td>74930</td>
      <td>74930</td>
      <td>74930</td>
      <td>74930</td>
      <td>74930</td>
      <td>74930.000000</td>
      <td>74930</td>
      <td>74930.000000</td>
      <td>7.493000e+04</td>
      <td>74930.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>50</td>
      <td>973</td>
      <td>58067</td>
      <td>2</td>
      <td>881</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>UTTAR PRADESH</td>
      <td>NALGONDA</td>
      <td>OM PRAKASH</td>
      <td>M</td>
      <td>IND</td>
      <td>NaN</td>
      <td>GEN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>15582</td>
      <td>575</td>
      <td>80</td>
      <td>71817</td>
      <td>41469</td>
      <td>NaN</td>
      <td>55949</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1989.949473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.774972</td>
      <td>NaN</td>
      <td>49700.452969</td>
      <td>9.573475e+05</td>
      <td>11.962859</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.981721</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.061647</td>
      <td>NaN</td>
      <td>94144.950189</td>
      <td>3.386265e+05</td>
      <td>30.865537</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1951.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1984.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.774972</td>
      <td>NaN</td>
      <td>940.000000</td>
      <td>7.228450e+05</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1991.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.774972</td>
      <td>NaN</td>
      <td>3535.000000</td>
      <td>9.573475e+05</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1998.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.774972</td>
      <td>NaN</td>
      <td>45776.750000</td>
      <td>1.154100e+06</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2009.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99.000000</td>
      <td>NaN</td>
      <td>855543.000000</td>
      <td>3.103525e+06</td>
      <td>480.000000</td>
    </tr>
  </tbody>
</table>
</div>



Taken care of missing values
This script will reformat all the names to the Format - (First Name Middle Name Last Name)


```python
def redo_names(x):
    n = x.replace(" ","")
    if n.count(",") == 1:
        lname, fname = n.split(',')
        return fname + " " + lname
    elif n.count(",") == 2:
        lname, fname, bname = n.split(",")
        return bname + " " + fname + " "+  lname
    elif n.count(",") == 3:
        return "AJIT KUMAR PANJA" #One off exceptional case
    else:
        return x

df["NAME"] = df["NAME"].map(redo_names)
```


```python
df.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>STATE</th>
      <th>PC</th>
      <th>NAME</th>
      <th>SEX</th>
      <th>PARTY</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>VOTES</th>
      <th>ELECTORS</th>
      <th>#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>JAWALA PRASHAD</td>
      <td>M</td>
      <td>INC</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>46679</td>
      <td>162327.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>CHAND KARAN</td>
      <td>M</td>
      <td>BJS</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>28990</td>
      <td>162327.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>DINO MAL</td>
      <td>M</td>
      <td>PURP</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>10778</td>
      <td>162327.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>BAJORIA BADRIDAS</td>
      <td>M</td>
      <td>IND</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>6153</td>
      <td>162327.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER NORTH</td>
      <td>RANG RAJ MEHTA</td>
      <td>M</td>
      <td>IND</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>4565</td>
      <td>162327.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER SOUTH</td>
      <td>MUKAT BEHARI LAL</td>
      <td>M</td>
      <td>INC</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>43082</td>
      <td>167157.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER SOUTH</td>
      <td>KUMARANAND</td>
      <td>M</td>
      <td>CPI</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>25128</td>
      <td>167157.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1951</td>
      <td>AJMER</td>
      <td>AJMER SOUTH</td>
      <td>MADAN SINGH</td>
      <td>M</td>
      <td>RRP</td>
      <td>45.774972</td>
      <td>ST</td>
      <td>13624</td>
      <td>167157.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1951</td>
      <td>ASSAM</td>
      <td>AUTONOMOUS DISTS</td>
      <td>BONILY KHONGMEN</td>
      <td>M</td>
      <td>INC</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>59326</td>
      <td>360630.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1951</td>
      <td>ASSAM</td>
      <td>AUTONOMOUS DISTS</td>
      <td>WILSON READE</td>
      <td>M</td>
      <td>KJD</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>32987</td>
      <td>360630.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



This dataset is now ready for analysis

1) Which year  has the most number of candidates?


```python
df.YEAR.value_counts()
```




    1996    13952
    1991     8668
    2009     8070
    1989     6160
    2004     5435
    1984     5312
    1998     4750
    1999     4648
    1980     4629
    1971     2784
    1977     2439
    1967     2369
    1962     1985
    1951     1874
    1957     1594
    1985      180
    1992       81
    Name: YEAR, dtype: int64




```python
## 2)Breakdown of male and female candidates
df.groupby(['YEAR', 'SEX']).size()
```




    YEAR  SEX
    1951  M       1874
    1957  M       1594
    1962  F         66
          M       1919
    1967  F         67
          M       2302
    1971  M       2784
    1977  F         70
          M       2369
    1980  F        143
          M       4486
    1984  F        162
          M       5150
    1985  F          9
          M        171
    1989  F        198
          M       5962
    1991  F        326
          M       8342
    1992  F          4
          M         77
    1996  F        599
          M      13353
    1998  F        274
          M       4476
    1999  F        284
          M       4364
    2004  F        355
          M       5080
    2009  F        556
          M       7514
    dtype: int64



3) Lets see the progression of sex ratio over Time



```python
c = df
c = c.groupby(['YEAR', 'SEX']).size()
c = c.unstack('SEX')
c = c.dropna()
print c.F/(c.M+c.F)
(c.F/(c.M+c.F)).plot(ylim = [0,0.5])

```

    YEAR
    1962    0.033249
    1967    0.028282
    1977    0.028700
    1980    0.030892
    1984    0.030497
    1985    0.050000
    1989    0.032143
    1991    0.037610
    1992    0.049383
    1996    0.042933
    1998    0.057684
    1999    0.061102
    2004    0.065317
    2009    0.068897
    dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x7f7351067090>




![png](/images/Election_files/output_17_2.png)



```python
#Its pathetically low
```

4)Which states has most and least candidates?


```python
df.STATE.value_counts().head(15).plot(kind = "bar")
plt.ylabel("Number of candidates")
plt.xlabel("States")
```




    <matplotlib.text.Text at 0x7f7351481cd0>




![png](/images/Election_files/output_20_1.png)



```python
df.STATE.value_counts().tail(15).plot(kind = "bar").invert_xaxis()
plt.ylabel("Number of Candidates")
```




    <matplotlib.text.Text at 0x7f73508d4610>




![png](/images/Election_files/output_21_1.png)


5) Which are the most & least competitive constituencies ?


```python
df.PC.value_counts().head(15).plot(kind = "bar")
plt.ylabel("Candidates")
```




    <matplotlib.text.Text at 0x7f7350731790>




![png](/images/Election_files/output_23_1.png)


6)Oldest and youngest candidates?


```python
c = df
c.sort_values("AGE").head(3)
#Youngest
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>STATE</th>
      <th>PC</th>
      <th>NAME</th>
      <th>SEX</th>
      <th>PARTY</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>VOTES</th>
      <th>ELECTORS</th>
      <th>#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73230</th>
      <td>2009</td>
      <td>UTTAR PRADESH</td>
      <td>AMETHI</td>
      <td>SWAMI NATH</td>
      <td>M</td>
      <td>IND</td>
      <td>25.0</td>
      <td>GEN</td>
      <td>9642</td>
      <td>1.431787e+06</td>
      <td>5</td>
    </tr>
    <tr>
      <th>66056</th>
      <td>2004</td>
      <td>UTTAR PRADESH</td>
      <td>MACHHLISHAHR</td>
      <td>ANIL KUMAR ALIAS AWADHESHANAND</td>
      <td>M</td>
      <td>IND</td>
      <td>25.0</td>
      <td>GEN</td>
      <td>4312</td>
      <td>9.573475e+05</td>
      <td>8</td>
    </tr>
    <tr>
      <th>66080</th>
      <td>2004</td>
      <td>UTTAR PRADESH</td>
      <td>MAINPURI</td>
      <td>RAKESH KUMAR S/O SHRI SHAITAN SINGH</td>
      <td>M</td>
      <td>IND</td>
      <td>25.0</td>
      <td>GEN</td>
      <td>966</td>
      <td>9.573475e+05</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
c = df
c.sort_values("AGE").tail(3)
#Oldest
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>STATE</th>
      <th>PC</th>
      <th>NAME</th>
      <th>SEX</th>
      <th>PARTY</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>VOTES</th>
      <th>ELECTORS</th>
      <th>#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67787</th>
      <td>2009</td>
      <td>BIHAR</td>
      <td>HAJIPUR</td>
      <td>RAM SUNDAR DAS</td>
      <td>M</td>
      <td>JD(U)</td>
      <td>88.0</td>
      <td>SC</td>
      <td>246715</td>
      <td>1.327075e+06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63100</th>
      <td>2004</td>
      <td>KARNATAKA</td>
      <td>BIDAR</td>
      <td>RAMCHANDRA VEERAPPA</td>
      <td>M</td>
      <td>BJP</td>
      <td>94.0</td>
      <td>SC</td>
      <td>312838</td>
      <td>9.573475e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>70704</th>
      <td>2009</td>
      <td>MAHARASHTRA</td>
      <td>DHULE</td>
      <td>ARIF AHMED SHAIKH JAFHAR</td>
      <td>M</td>
      <td>NBNP</td>
      <td>99.0</td>
      <td>GEN</td>
      <td>955</td>
      <td>1.575225e+06</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



7)Most common names in history


```python
df.mode()["NAME"][0]
```




    'OM PRAKASH'



8) Most common first name


```python
c = df["NAME"]
c = c.map(lambda x: x.split()[0])
c.mode()[0]
```




    'RAM'



9)Most common last name


```python
c = df["NAME"]
c = c.map(lambda x: x.split()[-1])
c.mode()[0]
```




    'SINGH'



10) Lets see the growth in number of votes every election


```python
df.groupby(["YEAR"]).sum()["VOTES"].plot(kind = "bar")
plt.ylabel("Votes x 100,000,00")
```




    <matplotlib.text.Text at 0x7f7350df2b90>




![png](/images/Election_files/output_34_1.png)


11)Top parties since 1951 and growth of number of parties


```python
c = df
c.groupby(['PARTY']).size().sort_values(ascending=False).head(20).drop("IND").plot(kind = "bar")
plt.ylabel("Number of Candidates")
```




    <matplotlib.text.Text at 0x7f73508c4750>




![png](/images/Election_files/output_36_1.png)


12)Independent Candidates in every election


```python
c = df
print "Number of IND Candidates =", c.groupby(['PARTY']).size()["IND"]
```

    Number of IND Candidates = 41469



```python
c = df
c[c.PARTY == "IND"]
c.groupby(["YEAR"]).size().plot(kind = "bar", title = "Number of Independent Candidates by Year")
plt.ylabel("Number of Candidates")
```




    <matplotlib.text.Text at 0x7f735083e3d0>




![png](/images/Election_files/output_39_1.png)


13)Percent of independent candidates every election


```python
c = df
d = df
c = c[c.PARTY == "IND"].groupby(["YEAR"]).size()
d = d[d.PARTY != "IND"].groupby(["YEAR"]).size()
f = (((c/d)*100).to_frame())
f.columns = ["% of Independent Candidates"]
f
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>% of Independent Candidates</th>
    </tr>
    <tr>
      <th>YEAR</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1951</th>
      <td>39.746458</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>51.520913</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>31.806109</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>57.618097</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>68.727273</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>100.740741</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>156.738769</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>249.243918</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>133.766234</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>151.736821</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>174.825618</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>65.306122</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>320.747889</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>67.548501</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>71.957085</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>78.196721</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>90.375088</td>
    </tr>
  </tbody>
</table>
</div>



14) Candidates who polled Maximum number of votes every election?


```python
d = (df.loc[(df.groupby(['YEAR'])['VOTES'].idxmax())])
d
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>STATE</th>
      <th>PC</th>
      <th>NAME</th>
      <th>SEX</th>
      <th>PARTY</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>VOTES</th>
      <th>ELECTORS</th>
      <th>#</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>462</th>
      <td>1951</td>
      <td>HYDERABAD</td>
      <td>NALGONDA</td>
      <td>RAVI NARAYAN REDDY</td>
      <td>M</td>
      <td>PDF</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>309162</td>
      <td>7.295040e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2256</th>
      <td>1957</td>
      <td>BOMBAY</td>
      <td>BOMBAY CITY CENTRAL</td>
      <td>DANGE SHRIPAD AMRIT</td>
      <td>M</td>
      <td>CPI</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>323526</td>
      <td>7.831160e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4362</th>
      <td>1962</td>
      <td>MAHARASHTRA</td>
      <td>BOMBAY CITY NORTH</td>
      <td>V. K. KRISHNA MENON</td>
      <td>M</td>
      <td>INC</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>298427</td>
      <td>7.640160e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7813</th>
      <td>1967</td>
      <td>WEST BENGAL</td>
      <td>TAMLUK</td>
      <td>S. C. SAMANTA</td>
      <td>M</td>
      <td>BAC</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>278623</td>
      <td>5.154150e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7891</th>
      <td>1971</td>
      <td>ANDHRA PRADESH</td>
      <td>KAKINADA</td>
      <td>M. S. SANJEEVI RAO</td>
      <td>M</td>
      <td>INC</td>
      <td>45.774972</td>
      <td>ST</td>
      <td>313060</td>
      <td>5.664610e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10946</th>
      <td>1977</td>
      <td>BIHAR</td>
      <td>HAJIPUR</td>
      <td>RAM VILAS PASWAN</td>
      <td>M</td>
      <td>BLD</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>469007</td>
      <td>6.869690e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17531</th>
      <td>1980</td>
      <td>WEST BENGAL</td>
      <td>DUM DUM</td>
      <td>NIREN GHOSH</td>
      <td>M</td>
      <td>CPM</td>
      <td>45.774972</td>
      <td>ST</td>
      <td>368214</td>
      <td>9.026110e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21409</th>
      <td>1984</td>
      <td>TAMIL NADU</td>
      <td>PUDUKKOTTAI</td>
      <td>SUNDARARAJ N.</td>
      <td>M</td>
      <td>INC</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>426717</td>
      <td>8.067520e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23012</th>
      <td>1985</td>
      <td>ASSAM</td>
      <td>GAUHATI</td>
      <td>DINESH GOSWAMI</td>
      <td>M</td>
      <td>IND</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>428013</td>
      <td>8.749810e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23716</th>
      <td>1989</td>
      <td>BIHAR</td>
      <td>HAJIPUR</td>
      <td>RAM VILAS PASWAN</td>
      <td>M</td>
      <td>JD</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>615129</td>
      <td>9.774990e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29430</th>
      <td>1991</td>
      <td>ANDHRA PRADESH</td>
      <td>CUDDAPAH</td>
      <td>Y.S. RAJASEKHAR REDDY</td>
      <td>M</td>
      <td>INC</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>583953</td>
      <td>1.138514e+06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38017</th>
      <td>1992</td>
      <td>PUNJAB</td>
      <td>GURDASPUR</td>
      <td>SUKHBUNS KAUR (W)</td>
      <td>F</td>
      <td>INC</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>172391</td>
      <td>9.640140e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46216</th>
      <td>1996</td>
      <td>NCT OF DELHI</td>
      <td>OUTER DELHI</td>
      <td>KRISHAN LAL SHARMA</td>
      <td>M</td>
      <td>BJP</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>701262</td>
      <td>2.821566e+06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54653</th>
      <td>1998</td>
      <td>NCT OF DELHI</td>
      <td>OUTER DELHI</td>
      <td>KRISHAN LAL SHARMA</td>
      <td>M</td>
      <td>BJP</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>715170</td>
      <td>2.926563e+06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59101</th>
      <td>1999</td>
      <td>NCT OF DELHI</td>
      <td>OUTER DELHI</td>
      <td>SAHIB SINGH VERMA</td>
      <td>M</td>
      <td>BJP</td>
      <td>45.774972</td>
      <td>GEN</td>
      <td>709692</td>
      <td>3.103525e+06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>64234</th>
      <td>2004</td>
      <td>NCT OF DELHI</td>
      <td>OUTER DELHI</td>
      <td>SAJJAN KUMAR</td>
      <td>M</td>
      <td>INC</td>
      <td>58.000000</td>
      <td>GEN</td>
      <td>855543</td>
      <td>9.573475e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71357</th>
      <td>2009</td>
      <td>NAGALAND</td>
      <td>NAGALAND</td>
      <td>C.M. CHANG</td>
      <td>M</td>
      <td>NPF</td>
      <td>65.000000</td>
      <td>ST</td>
      <td>832224</td>
      <td>1.321878e+06</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
