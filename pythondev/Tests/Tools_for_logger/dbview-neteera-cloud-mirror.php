<html>
<head>
    <title>Neteera's Database</title>
    <!-- Bootstrap CDN -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
</head>
<style>

.pagination {
  display: inline-block;
}


.pagination a {
  color: black;
  float: center;
  padding: 8px 16px;
  text-decoration: none;
  transition: background-color .3s;
}

.pagination a.active {
  background-color: #4CAF50;
  color: white;
}

.pagination a:hover:not(.active) {background-color: #ddd;}

table, th, td, tr {
    border: 2px solid black;
    margin-top: 50px;
    margin-right: 5px;

}

th {
  text-align: center;
  font-size: 17px;
  background-color: lightblue;
}

tr:hover {background-color: #f5f5d5;}

td :nth-child(even){background-color: #f5f5d5;}

td {
  text-align: center;
  font-size: 13px;
  font-weight: bold;
  color: black;
}
a.button {
display: inline-block;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
  text-align: center;
  text-decoration: none;
  outline: none;
  color: black;
  background-color: lightblue;
  border: none;
  border-radius: 10px;
  box-shadow: 0 9px #999;
}

.button:hover {background-color: #f5f5d5}

.button:active {
  background-color: #3e8e41;
  box-shadow: 0 5px #666;
  transform: translateY(4px);
}


</style>
<body>
<a class="button" href=../dbview-cpd.php><input type=button value='CPD-View'></a>


<a class="button" href=../dbview-full.php><input type=button value='Full-view-neteera_db'></a>


<?php

$host    = "localhost";
$user    = "user";
$pass    = "1234";
$db_name = "neteera_cloud_mirror";

//create connection
$conn = mysqli_connect($host, $user, $pass, $db_name);



$showRecordPerPage = 99999;
if(isset($_GET['page']) && !empty($_GET['page'])){
$currentPage = $_GET['page'];
}else{
$currentPage = 1;
}
$startFrom = ($currentPage * $showRecordPerPage) - $showRecordPerPage;
$totalEmpSQL = "SELECT * FROM session";
$allEmpResult = mysqli_query($conn, $totalEmpSQL);
$totalEmployee = mysqli_num_rows($allEmpResult);
$lastPage = ceil($totalEmployee/$showRecordPerPage);
$firstPage = 1;
$nextPage = $currentPage + 1;
$previousPage = $currentPage - 1;
$empSQL = "SELECT Session, Setup, Timestamp, Duration, Path, Company, Subject, Posture, Location, Target, FW, Model, Distance, SN, Validity, Notes, ss.event_comment


FROM session ss

         INNER JOIN
     ( SELECT ss.session_id AS Session,
              GROUP_CONCAT(DISTINCT se.setup_id ORDER BY se.setup_id SEPARATOR ', ') AS Setup,
              GROUP_CONCAT(DISTINCT se.legacy_id ORDER BY se.setup_id SEPARATOR ', ') AS Legacy,
              ss.date AS Timestamp,
              ss.duration_sec AS Duration,
              CONCAT(DATE_FORMAT(date(ss.date), '%Y\%c\%e'), '\\\\', se.fk_session_id) AS Path,
              su.name AS Subject,
              sp.name AS Posture,
              su2.name AS Operator,
              c.name AS Company,
              ss.note AS Notes
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id
                LEFT JOIN subject su
                          ON ss.fk_subject_id = su.subject_id
                LEFT JOIN join_session_scenario jss
                          ON ss.session_id = jss.fk_session_id
                LEFT JOIN scenario sc
                          ON jss.fk_scenario_id = sc.scenario_id
                LEFT JOIN neteera_staff_operator op
                          ON ss.fk_neteera_staff_operator_id = op.neteera_staff_operator_id
                LEFT JOIN subject su2
                          ON op.fk_subject_id = su2.subject_id
                LEFT JOIN subject_posture sp
                          ON ss.fk_subject_posture_id = sp.subject_posture_id
                LEFT JOIN company c
                          ON ss.fk_company_id = c.company_id
       WHERE ss.session_id > 3800
       GROUP BY session_id
     ) AS SESSION_DATA
     ON ss.session_id = SESSION_DATA.Session


         LEFT JOIN
     (SELECT ss.session_id,
             GROUP_CONCAT(DISTINCT fp.name ORDER BY fp.name SEPARATOR ', ') AS GT_POS
      FROM session ss
               JOIN setup se
                    ON ss.session_id = se.fk_session_id
               LEFT JOIN setup_type st
                         ON se.fk_setup_type_id = st.setup_type_id
               LEFT JOIN data dt
                         ON se.fk_session_id = dt.fk_session_id
               LEFT JOIN reference_data_hist rd
                         ON dt.data_id = rd.fk_data_id
               LEFT JOIN feature_param fp
                         ON rd.fk_feature_param_id = fp.feature_param_id
               LEFT JOIN feature_param_type t
                         ON fp.fk_feature_param_type_id = t.feature_param_type_id
               LEFT JOIN gt_recorder gt
                         ON fp.feature_param_id = gt.fk_feature_param_id
      WHERE  t.name = 'gt' AND rd.bin = 1
      GROUP BY session_id
     ) AS SESSION_GT_POS
     ON ss.session_id = SESSION_GT_POS.session_id


         LEFT JOIN
     (SELECT ss.session_id,
             GROUP_CONCAT(DISTINCT gt.opposite ORDER BY gt.opposite SEPARATOR ', ') AS GT_NEG
      FROM session ss
               JOIN setup se
                    ON ss.session_id = se.fk_session_id
               LEFT JOIN setup_type st
                         ON se.fk_setup_type_id = st.setup_type_id
               LEFT JOIN data dt
                         ON se.fk_session_id = dt.fk_session_id
               LEFT JOIN reference_data_hist rd
                         ON dt.data_id = rd.fk_data_id
               LEFT JOIN feature_param fp
                         ON rd.fk_feature_param_id = fp.feature_param_id
               LEFT JOIN feature_param_type t
                         ON fp.fk_feature_param_type_id = t.feature_param_type_id
               LEFT JOIN gt_recorder gt
                         ON fp.feature_param_id = gt.fk_feature_param_id
      WHERE  t.name = 'gt' AND rd.bin = 0
      GROUP BY session_id
     ) AS SESSION_GT_NEG
     ON ss.session_id = SESSION_GT_NEG.session_id



         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(DISTINCT f.name ORDER BY f.name SEPARATOR ', ') as State
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id

                LEFT JOIN join_session_feature_param jsf
                          ON ss.session_id = jsf.fk_session_id
                LEFT JOIN feature_param f
                          ON jsf.fk_feature_param_id = f.feature_param_id
                LEFT JOIN feature_param_type t
                          ON f.fk_feature_param_type_id = t.feature_param_type_id
       WHERE  t.name = 'state' AND jsf.value = 'TRUE'
       GROUP BY session_id
     ) AS SESSION_STATE
     ON ss.session_id = SESSION_STATE.session_id


         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(sw.name ORDER BY se.setup_id SEPARATOR ', ') AS Version
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id
                LEFT JOIN setup_type st
                          ON se.fk_setup_type_id = st.setup_type_id
                LEFT JOIN test te
                          ON se.setup_id = te.fk_setup_id
                LEFT JOIN sw_version sw
                          ON te.fk_sw_version_id = sw.sw_version_id
       WHERE st.name != 'OCC' AND ((te.res_fpath LIKE '%usb_results%' AND te.res_fpath NOT LIKE '%NES_results%') OR
                                   (te.res_fpath NOT LIKE '%usb_results%' AND te.res_fpath LIKE '%NES_results%') OR
                                   (te.res_fpath LIKE '%mqtt_results%' AND te.res_fpath NOT LIKE '%usb_results%' AND te.res_fpath NOT LIKE '%NES_results%'))
       GROUP BY session_id
     ) AS SESSION_VERSION
     ON ss.session_id = SESSION_VERSION.session_id


         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(mu.name ORDER BY se.setup_id SEPARATOR ', ') AS Location
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id

                LEFT JOIN nes_mount_area mu
                          ON se.fk_nes_mount_area_id = mu.nes_mount_area_id

       GROUP BY session_id
     ) AS SETUP_LOCATION
     ON ss.session_id = SETUP_LOCATION.session_id


         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(nsp.name ORDER BY se.setup_id SEPARATOR ', ') AS Target
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id

                LEFT JOIN nes_subject_position nsp
                          ON se.fk_nes_subject_position_id = nsp.nes_subject_position_id

       GROUP BY session_id
     ) AS SETUP_TARGET
     ON ss.session_id = SETUP_TARGET.session_id

         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(jnc.value ORDER BY se.setup_id SEPARATOR ', ') AS FW
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id

                LEFT JOIN nes_config nc
                          ON se.fk_nes_config_id = nc.nes_config_id
                LEFT JOIN join_nes_config_param jnc
                          ON nc.nes_config_id = jnc.fk_nes_config_id
                LEFT JOIN nes_param fw
                          ON jnc.fk_nes_param_id = fw.nes_param_id
       WHERE  fw.name = 'infoFW_version'
       GROUP BY session_id
     ) AS SETUP_FW
     ON ss.session_id = SETUP_FW.session_id


         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(nm.name ORDER BY se.setup_id SEPARATOR ', ') AS Model
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id

                LEFT JOIN nes_config nc
                          ON se.fk_nes_config_id = nc.nes_config_id
                LEFT JOIN nes_model nm
                          ON nc.fk_nes_model_id = nm.nes_model_id

       GROUP BY session_id
     ) AS SETUP_MODEL
     ON ss.session_id = SETUP_MODEL.session_id


         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(se.target_distance_mm ORDER BY se.setup_id SEPARATOR ', ') AS Distance
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id


       GROUP BY session_id
     ) AS SETUP_DISTANCE
     ON ss.session_id = SETUP_DISTANCE.session_id


         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(DISTINCT ci.serial_num ORDER BY se.setup_id SEPARATOR ', ') AS SN
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id

                LEFT JOIN nes_config nc
                          ON se.fk_nes_config_id = nc.nes_config_id
                LEFT JOIN join_nes_config_component_inventory jcc
                          ON nc.nes_config_id = jcc.fk_nes_config_id
                LEFT JOIN component_inventory ci
                          ON jcc.fk_component_inventory_id = ci.component_inventory_id

       GROUP BY session_id
     ) AS SETUP_SN
     ON ss.session_id = SETUP_SN.session_id


         LEFT JOIN
     ( SELECT ss.session_id,
              GROUP_CONCAT(dv.name ORDER BY se.setup_id SEPARATOR ', ') AS Validity
       FROM session ss
                JOIN setup se
                     ON ss.session_id = se.fk_session_id
                LEFT JOIN data d
                          ON se.setup_id = d.fk_setup_id
                LEFT JOIN data_validity dv
                          ON d.fk_data_validity_id = dv.data_validity_id
                LEFT JOIN sensor sn
                          ON d.fk_sensor_id = sn.sensor_id
       WHERE  sn.name = 'NES' AND (LENGTH(d.fpath)-LENGTH(REPLACE(d.fpath,'.','')))/LENGTH('.') = 1 AND ((substring_index(d.fpath, '.', -1) = 'ttlog') OR (substring_index(d.fpath, '.', -1) = 'tlog') OR (substring_index(d.fpath, '.', -1) = 'blog'))
       GROUP BY session_id
     ) AS SETUP_NES_VALIDITY
     ON ss.session_id = SETUP_NES_VALIDITY.session_id
     group by ss.session_id DESC";
$empResult = mysqli_query($conn, $empSQL);
?>
<table border:' 2px solid black' >
<thead>
<tr>
<th>Session</th>
<th>Setup</th>
<th>Timestamp</th>
<th>Duration</th>
<th>Path</th>
<th>Company</th>
<th>Subject</th>
<th>Posture</th>
<th>Location</th>
<th>Target</th>
<th>FW</th>
<th>Model</th>
<th>Distance</th>
<th>SN</th>
<th>Validity</th>
<th>Event Comment</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<?php
while($emp = mysqli_fetch_assoc($empResult)){
?>
<tr>
<th scope="row"><?php echo $emp['Session']; ?></th>
<td><?php echo $emp['Setup']; ?></td>
<td><?php echo $emp['Timestamp']; ?></td>
<td><?php echo $emp['Duration']; ?></td>
<td><a href=" <?php echo  "file:////nas.neteera.local/data/", $emp['Path'];?>">path</a></td>
<td><?php echo $emp['Company']; ?></td>
<td><?php echo $emp['Subject']; ?></td>
<td><?php echo $emp['Posture']; ?></td>
<td><?php echo $emp['Location']; ?></td>
<td><?php echo $emp['Target']; ?></td>
<td><?php echo $emp['FW']; ?></td>
<td><?php echo $emp['Model']; ?></td>
<td><?php echo $emp['Distance']; ?></td>
<td><?php echo $emp['SN']; ?></td>
<td><?php echo $emp['Validity']; ?></td>
<td><?php echo $emp['event_comment']; ?></td>
<td><?php echo $emp['Notes']; ?></td>
</tr>
<?php } ?>
</tbody>
</table>
<nav aria-label="Page navigation">
<div class="text-center">
<ul class="pagination pagination-lg">
<?php if($currentPage != $firstPage) { ?>
<li class="page-item"><a class="page-link" href="?page=<?php echo $firstPage ?>" tabindex="-1" aria-label="Previous"><span aria-hidden="true">First</span></a></li>
<?php } ?>
<?php if($currentPage >= 2) { ?>
<li class="page-item"><a class="page-link" href="?page=<?php echo $previousPage ?>"><?php echo $previousPage ?></a></li>
<?php } ?>
<li class="page-item active"><a class="page-link" href="?page=<?php echo $currentPage ?>"><?php echo $currentPage ?></a></li>
<?php if($currentPage != $lastPage) { ?>
<li class="page-item"><a class="page-link" href="?page=<?php echo $nextPage ?>"><?php echo $nextPage ?></a></li>
<li class="page-item"><a class="page-link" href="?page=<?php echo $lastPage ?>" aria-label="Next"><span aria-hidden="true">Last</span></a></li>
<?php } ?>
</ul>
</div>
</nav>
</body>
</html>
